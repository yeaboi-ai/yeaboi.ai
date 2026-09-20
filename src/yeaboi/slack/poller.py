"""Read Slack back on a cadence, and apply what an allowlisted human said.

yeaboi has no daemon and this does not give it one. A short job fires every few
minutes, reads a fixed window, applies what it finds and exits — which is the
same process model the ceremonies clock already uses, and it is why the whole
inbound-request security surface is absent: there is no public endpoint to sign,
no ``X-Slack-Signature`` to verify, no timestamp replay window and no listening
socket. yeaboi opens every connection; TLS authenticates Slack and the bot token
authenticates yeaboi.

**A fixed 48-hour window, not a cursor.** Runs overlap on purpose. A cursor
means a gap after any failed run, and that gap silently drops somebody's vote;
an overlap costs one extra API call and nothing else, because the ledger's
primary key makes a replay free. The fleet's relay reasoned its way to the same
window and then kept its marker *in Slack* — that part is not copied here, and
the reason is Slack's own documented behaviour: ``reactions[].users`` truncates
at about fifty per emoji, so on a busy message the marker reads as absent and
the act fires twice. A primary key cannot be truncated.

# See docs: "Integrations" — Slack
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from yeaboi.slack import allowlist as allow
from yeaboi.slack import grammar
from yeaboi.slack.apply import ApplyResult
from yeaboi.slack.store import (
    OUTCOME_APPLIED,
    OUTCOME_FAILED,
    OUTCOME_IGNORED,
    OUTCOME_REFUSED,
    OUTCOME_STALE,
    OUTCOME_UNAUTHORIZED,
    POLL_FAILED,
    POLL_LOCKED,
    POLL_NO_ALLOWLIST,
    POLL_NO_CHANNEL,
    POLL_NO_TOKEN,
    POLL_OK,
    InboundEvent,
    SlackStore,
    reaction_key,
    reply_key,
)
from yeaboi.timeparse import parse_datetime

logger = logging.getLogger(__name__)

#: How far back every poll reads, regardless of when the last one ran.
WINDOW_HOURS = 48

#: Message subtypes that are still an ordinary human message.
_HUMAN_SUBTYPES = frozenset({"", "thread_broadcast"})


@dataclass(frozen=True)
class PollResult:
    """What one poll did. Mirrors the row it writes."""

    outcome: str = POLL_OK
    messages_read: int = 0
    events_seen: int = 0
    events_new: int = 0
    events_applied: int = 0
    duration_s: float = 0.0
    detail: str = ""
    error: str = ""

    @property
    def declined(self) -> bool:
        return self.outcome.startswith("skipped_")


def is_human_message(message: dict) -> bool:
    """True only for a plain, first-party, human-authored message.

    Everything here fails closed. ``username`` is client-settable, so it is
    never a fallback for a missing ``user``; a ``bot_id`` or ``app_id`` means
    another integration is talking, and an unexpected subtype
    (``message_changed``, ``message_deleted``, a file share with no author)
    means the shape is not the one this grammar was written against.
    """
    if not isinstance(message, dict):
        return False
    if message.get("bot_id") or message.get("app_id"):
        return False
    if str(message.get("subtype", "")) not in _HUMAN_SUBTYPES:
        return False
    return bool(str(message.get("user", "")).strip())


def _window_start(now: datetime) -> datetime:
    return now - timedelta(hours=WINDOW_HOURS)


def _lock(path: Path):
    """A best-effort exclusive lock, or None when someone else holds it.

    launchd will not run a second instance of a loaded label, but **cron
    absolutely will** — and two polls racing the same window would both claim
    different halves of it. The kernel drops the lock when the process dies, so
    there is no stale-lock case to recover from.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover — Windows has no scheduling here anyway
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def collect_events(
    store: SlackStore, api, *, channel: str, token: str, allowed: tuple[str, ...], bot_id: str, now: datetime
) -> tuple[list[InboundEvent], int, str]:
    """Read the window and turn it into events. Returns (events, anchors_read, error).

    Reads only what yeaboi itself posted: the anchors are the index, so a
    channel full of other traffic costs nothing and can say nothing.
    """
    oldest = _window_start(now).isoformat(timespec="seconds")
    anchors = store.anchors_since(oldest)
    # Posts are the unit of reading: a signal anchor is reached through the
    # thread of the post it hangs under, so iterating it separately would spend
    # one `reactions.get` per signal — thirteen calls where two will do.
    posts = [a for a in anchors if not a.is_signal]
    events: list[InboundEvent] = []
    error = ""

    for anchor in posts:
        resp = api.reactions_get(anchor.channel, anchor.ts, token=token)
        if not resp.ok:
            if api.is_fatal_auth_error(resp):
                return events, len(anchors), resp.error
            # One unreadable message must not cost the rest of the window.
            logger.warning("slack: could not read reactions on %s (%s)", anchor.ts, resp.error)
            error = error or resp.error
            continue
        events.extend(_reaction_events((resp.data.get("message") or {}).get("reactions"), anchor))

        thread, err = api.paginate(
            lambda cursor, a=anchor: api.replies(a.channel, a.ts, cursor=cursor, token=token), "messages"
        )
        if err:
            logger.warning("slack: could not read the thread on %s (%s)", anchor.ts, err)
            error = error or err
            continue

        # Our own signal replies, by ts. Reading their reactions out of the
        # thread we already fetched is what makes a votable standup cost no
        # extra API calls at all.
        signals = {s.ts: s for s in store.thread(anchor.channel, anchor.ts)}
        for message in thread:
            reply_ts = str(message.get("ts", ""))
            if reply_ts == anchor.ts:
                continue
            signal = signals.get(reply_ts)
            if signal is not None:
                events.extend(_signal_events(message, signal, api, token=token))
                continue
            if not is_human_message(message):
                continue
            # Slack threads are flat — every reply carries `thread_ts =
            # root_ts` — so typed text always arrives against the post, never
            # against one of the signal replies above it. A reaction is
            # per-message and can tell them apart; text cannot.
            act, intent, payload = grammar.parse_reply(str(message.get("text", "")))
            events.append(
                InboundEvent(
                    event_key=reply_key(anchor.channel, reply_ts),
                    channel=anchor.channel,
                    anchor_ts=anchor.ts,
                    reply_ts=reply_ts,
                    act=act,
                    intent=intent,
                    payload=payload,
                    slack_user=str(message.get("user", "")),
                    anchor=anchor,
                )
            )

    _ = (allowed, bot_id)  # authorisation happens per event, in run_poll
    return events, len(anchors), error


def _reaction_events(reactions, anchor) -> list[InboundEvent]:
    """Every reaction on one of our messages, as events."""
    events: list[InboundEvent] = []
    for reaction in reactions or []:
        emoji = grammar.normalise_emoji(str(reaction.get("name", "")))
        act, intent = grammar.parse_reaction(emoji, on_signal=anchor.is_signal)
        for actor in reaction.get("users") or []:
            events.append(
                InboundEvent(
                    event_key=reaction_key(anchor.channel, anchor.ts, actor, emoji),
                    channel=anchor.channel,
                    anchor_ts=anchor.ts,
                    act=act,
                    intent=intent,
                    slack_user=actor,
                    anchor=anchor,
                )
            )
    return events


def _signal_events(message: dict, signal, api, *, token: str) -> list[InboundEvent]:
    """Verdicts on one signal reply, escalating only when the list is short.

    ``conversations.replies`` returns each message's reactions inline but
    truncates ``users`` at ~25, and a truncated list is indistinguishable from a
    short one — so a vote would go missing with nothing to notice. It also
    returns ``count``, which makes truncation *detectable*: ask
    ``reactions.get`` for the full list exactly when the two disagree, and never
    otherwise.
    """
    reactions = message.get("reactions") or []
    if any(int(r.get("count", 0) or 0) > len(r.get("users") or []) for r in reactions):
        resp = api.reactions_get(signal.channel, signal.ts, token=token)
        if resp.ok:
            reactions = (resp.data.get("message") or {}).get("reactions") or []
        else:
            logger.warning("slack: could not read the full reactions on signal %s (%s)", signal.ts, resp.error)
    return _reaction_events(reactions, signal)


def run_poll(
    *,
    db_path: Path | None = None,
    now: datetime | None = None,
    api=None,
    apply_event=None,
) -> PollResult:
    """One pass: read the window, apply what is new, record what happened.

    Never raises. Every outcome — including every decline — writes a ledger row,
    because a job firing unattended every few minutes and quietly stopping is
    indistinguishable from one that had nothing to do.
    """
    from yeaboi import config
    from yeaboi.logging_setup import mode_log

    moment = now or datetime.now(timezone.utc)
    started = time.monotonic()
    api = api or _default_api()
    if apply_event is None:
        from yeaboi.slack.apply import apply_event as _apply

        apply_event = _apply

    def _finish(result: PollResult) -> PollResult:
        result = replace(result, duration_s=round(time.monotonic() - started, 2))
        try:
            with SlackStore(db_path) as store:
                store.record_poll(
                    {
                        "polled_at": moment.isoformat(timespec="seconds"),
                        "window_start": _window_start(moment).isoformat(timespec="seconds"),
                        "outcome": result.outcome,
                        "messages_read": result.messages_read,
                        "events_seen": result.events_seen,
                        "events_new": result.events_new,
                        "events_applied": result.events_applied,
                        "duration_s": result.duration_s,
                        "detail": result.detail,
                        "error": result.error,
                    }
                )
        except Exception:  # noqa: BLE001 — the ledger is the record, not the run
            logger.warning("slack: could not record the poll", exc_info=True)
        logger.info("slack poll %s: %s", result.outcome, result.detail or "-")
        return result

    # Its own log file rather than the ceremonies one, for the reason that one
    # is separate from each mode's: this fires unattended on its own cadence,
    # and "did anyone's reaction get read?" must not be a question you answer
    # by reading around the runs a human started.
    #
    # It opens BEFORE the token/channel/allowlist/lock checks, not after. Those
    # four declines and the outer handler's stack trace are precisely the
    # diagnostics this directory exists for, and every one of them returned
    # through _finish() before the handler was ever installed.
    with mode_log("slack"):
        token = config.get_slack_bot_token()
        if not token:
            return _finish(PollResult(outcome=POLL_NO_TOKEN, detail="no SLACK_BOT_TOKEN — a webhook cannot read"))
        channel = config.get_slack_channel_id()
        if not channel:
            return _finish(PollResult(outcome=POLL_NO_CHANNEL, detail="no SLACK_CHANNEL_ID"))

        allowed = allow.load()
        if not allowed:
            # Not even a request: with nobody authorised there is no event this poll
            # could act on, so calling Slack would be pure waste.
            return _finish(PollResult(outcome=POLL_NO_ALLOWLIST, detail=allow.describe()))

        from yeaboi.paths import get_slack_log_dir

        handle = _lock(get_slack_log_dir() / "poll.lock")
        if handle is None:
            return _finish(PollResult(outcome=POLL_LOCKED, detail="another poll is already running"))

        try:
            identity = api.auth_test(token=token)
            if not identity.ok:
                return _finish(
                    PollResult(outcome=POLL_FAILED, error=identity.error, detail=api.error_message(identity))
                )
            bot_id = str(identity.data.get("user_id", ""))

            with SlackStore(db_path) as store:
                events, anchors_read, read_error = collect_events(
                    store, api, channel=channel, token=token, allowed=allowed, bot_id=bot_id, now=moment
                )
                if read_error and not events:
                    return _finish(
                        PollResult(
                            outcome=POLL_FAILED,
                            messages_read=anchors_read,
                            error=read_error,
                            detail=_help_for(api, read_error),
                        )
                    )

                new = applied = 0
                for event in events:
                    result = _handle(
                        store,
                        event,
                        allowed=allowed,
                        bot_id=bot_id,
                        apply_event=apply_event,
                        now=moment,
                        db_path=db_path,
                    )
                    if result is None:
                        continue
                    new += 1
                    if result.applied:
                        applied += 1
                        _ack(api, event, channel=event.channel, token=token)
                    # A deferral earns both a ✅ and a line, because "recorded,
                    # but later than you expected" is not something a tick can
                    # say on its own.
                    if result.speak and result.detail:
                        _say(api, event, result.detail, channel=event.channel, token=token)

                # Read the gap BEFORE this poll's own row lands, and prune after
                # — otherwise the comparison is against ourselves.
                gap = _gap_notice(store, moment)
                store.prune(now=moment)

            return _finish(
                PollResult(
                    outcome=POLL_OK,
                    messages_read=anchors_read,
                    events_seen=len(events),
                    events_new=new,
                    events_applied=applied,
                    detail=gap or (f"{applied} applied" if applied else "nothing new"),
                    error=read_error,
                )
            )
        except Exception as exc:  # noqa: BLE001 — an unattended job records rather than crashes
            logger.error("slack poll failed", exc_info=True)
            return _finish(PollResult(outcome=POLL_FAILED, error=f"{type(exc).__name__}: {exc}"))
        finally:
            handle.close()


def _handle(store, event, *, allowed, bot_id, apply_event, now, db_path=None) -> ApplyResult | None:
    """Claim and settle one event. None when a previous poll already had it.

    The returned ``outcome`` is always the settled word, so the caller never
    re-derives it — and ``speak``/``applied`` are what it uses to decide the ✅
    and the thread line, which are two different questions: a deferral earns
    both, an unauthorised reaction earns neither.

    ``db_path`` goes to the applier for the same reason every other store here
    takes it: without it a poll reads its anchors and writes its ledger in one
    database while pausing a ceremony and casting a verdict in another, and the
    claim row saying the event was handled lands on the wrong side — so the
    same window replayed against the real database would apply it twice.
    """
    anchor = event.anchor
    # Authorisation first, and silently: the channel is not the place to
    # announce who is unauthorised, and a bot that answers unknown users is one
    # anybody in the channel can make spam it.
    if not allow.authorised(event.slack_user, allowed, bot_user_id=bot_id):
        outcome, reason = OUTCOME_UNAUTHORIZED, "not on SLACK_ALLOWED_MEMBER_IDS"
    elif not event.act:
        outcome, reason = OUTCOME_IGNORED, "not part of the grammar"
    elif anchor is not None and anchor.expired(now):
        outcome, reason = OUTCOME_STALE, f"the post from {anchor.posted_at} is past its answering window"
    else:
        outcome = ""
        reason = ""

    if not store.claim(event, now=now):
        return None  # an earlier overlapping window already had this one

    if outcome:
        store.settle(event.event_key, outcome=outcome, reason=reason, now=now)
        return ApplyResult(False, reason, outcome=outcome)

    try:
        result = apply_event(event, db_path=db_path)
    except Exception as exc:  # noqa: BLE001 — one bad event must not stop the rest
        logger.error("slack: applying %s raised", event.event_key, exc_info=True)
        reason = f"{type(exc).__name__}: {exc}"
        store.settle(event.event_key, outcome=OUTCOME_FAILED, reason=reason, now=now)
        return ApplyResult(False, reason, outcome=OUTCOME_FAILED)

    settled = result.outcome or (OUTCOME_APPLIED if result.applied else OUTCOME_REFUSED)
    store.settle(event.event_key, outcome=settled, reason=result.detail, now=now)
    return replace(result, outcome=settled)


def _ack(api, event, *, channel: str, token: str) -> None:
    """Tick the message, when the token was given permission to.

    Off unless ``SLACK_ACK_REACTION`` is set: this is a read feature, and
    ``reactions:write`` is the scope an administrator is most likely to refuse.
    It is a courtesy for humans and never read back as a record, so failing to
    add it is logged and forgotten.
    """
    from yeaboi import config

    emoji = config.get_slack_ack_reaction()
    if not emoji:
        return
    ts = event.reply_ts or event.anchor_ts
    resp = api.add_reaction(channel, ts, emoji, token=token)
    if not resp.ok:
        logger.info("slack: could not tick %s (%s)", ts, resp.error)


def _say(api, event, text: str, *, channel: str, token: str) -> None:
    """Answer in the thread, for the few outcomes a ✅ cannot express.

    Always on, unlike :func:`_ack`, because it needs no scope the bot does not
    already hold — ``chat:write`` is what posted the dispatch in the first
    place. Restraint is in *what* speaks, not in whether it can: the caller
    sends only the results that asked to be spoken, so an unauthorised actor,
    an out-of-grammar reaction and an act a later phase will build all stay
    silent.
    """
    root = event.anchor.root_ts if event.anchor is not None else ""
    resp = api.post_message(channel, text, thread_ts=root or event.anchor_ts, token=token)
    if not resp.ok:
        logger.info("slack: could not answer in the thread on %s (%s)", event.anchor_ts, resp.error)


def _gap_notice(store, now: datetime) -> str:
    """Say so when we were away longer than the window can cover.

    Deliberately not widening the window to catch up: acting on a three-day-old
    approval is exactly the staleness the ceremonies engine already ruled
    against, and doing it silently is worse than saying it was missed.
    """
    last = store.last_poll(ok_only=True)
    if not last or not last.get("polled_at"):
        return ""
    try:
        previous = parse_datetime(last["polled_at"])
    except ValueError:
        return ""
    hours = (now - previous).total_seconds() / 3600
    if hours <= WINDOW_HOURS:
        return ""
    return f"gap of {hours:.0f}h exceeds the {WINDOW_HOURS}h window — anything older than that was not read"


def _help_for(api, error: str) -> str:
    """The actionable message for a Slack error code, without a response to hand."""
    from yeaboi.tools.slack import SlackResponse

    return api.error_message(SlackResponse(ok=False, error=error))


def _default_api():
    from yeaboi.tools import slack as slack_api

    return slack_api
