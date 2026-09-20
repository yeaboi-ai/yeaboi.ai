"""The poll: read a window, apply what an allowlisted human asked for, exit.

The properties under test are the ones an unattended job lives or dies on.
**Replay is free** — the window overlaps by design, so the same reaction seen
288 times must act once. **Failure is closed** — no token, no allowlist, a
malformed allowlist and an expired anchor all end in nothing happening, and all
of them say so in the ledger. And **one bad event never costs the rest**.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from yeaboi.slack import poller
from yeaboi.slack.apply import ApplyResult
from yeaboi.slack.poller import PollResult, is_human_message, run_poll
from yeaboi.slack.store import (
    KIND_SIGNAL,
    POLL_LOCKED,
    POLL_NO_ALLOWLIST,
    POLL_NO_CHANNEL,
    POLL_NO_TOKEN,
    POLL_OK,
    SlackAnchor,
    SlackStore,
)
from yeaboi.tools.slack import SlackResponse

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
ACTOR = "U0123456789"


@pytest.fixture
def db(tmp_path):
    return tmp_path / "sessions.db"


@pytest.fixture(autouse=True)
def _configured(monkeypatch, tmp_path):
    # Retention uses the poll's fixed clock, including when the store prunes.
    monkeypatch.setattr("yeaboi.slack.store._now", lambda: NOW)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-1")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")
    monkeypatch.setenv("SLACK_ALLOWED_MEMBER_IDS", ACTOR)
    monkeypatch.setenv("SLACK_ACK_REACTION", "")
    monkeypatch.setattr("yeaboi.paths.get_slack_log_dir", lambda: tmp_path)


class FakeApi:
    """A Slack that answers from a script rather than a network."""

    is_fatal_auth_error = staticmethod(lambda resp: resp.error in {"invalid_auth", "token_revoked"})
    error_message = staticmethod(lambda resp: f"Error: {resp.error}")

    def __init__(self, *, reactions=None, thread=None, auth=None):
        self._reactions = reactions or {}
        self._thread = thread or {}
        self._auth = auth or SlackResponse(ok=True, data={"user_id": "UBOT000000", "team": "acme", "user": "yeaboi"})
        self.acks: list[tuple[str, str, str]] = []
        self.said: list[tuple[str, str]] = []

    def auth_test(self, *, token="", budget=None):
        return self._auth

    def reactions_get(self, channel, ts, *, token="", budget=None):
        entry = self._reactions.get(ts)
        if isinstance(entry, SlackResponse):
            return entry
        return SlackResponse(ok=True, data={"message": {"reactions": entry or []}})

    def replies(self, channel, ts, *, cursor="", limit=200, token="", budget=None):
        return SlackResponse(ok=True, data={"messages": self._thread.get(ts, [])})

    def paginate(self, fetch, key, *, max_pages=10):
        resp = fetch("")
        return (list(resp.data.get(key) or []), "") if resp.ok else ([], resp.error)

    def add_reaction(self, channel, ts, name, *, token="", budget=None):
        self.acks.append((channel, ts, name))
        return SlackResponse(ok=True)

    def post_message(self, channel, text, *, thread_ts="", token="", budget=None):
        self.said.append((thread_ts, text))
        return SlackResponse(ok=True, data={"channel": channel, "ts": "9999.0001"})


def _anchor(db, **kw) -> SlackAnchor:
    base = {
        "channel": "C123",
        "ts": "1723800000.000100",
        "session_id": "s1",
        "ceremony": "morning",
        "mode": "standup",
        "run_id": 7,
    }
    with SlackStore(db) as store:
        return store.record_anchor(SlackAnchor(**{**base, **kw}), now=NOW - timedelta(hours=1))


def _react(emoji=None, users=None):
    return {"1723800000.000100": [{"name": emoji or "pause_button", "users": users or [ACTOR]}]}


def _human_reply(text="Ada was on leave, that ticket is not hers", ts="1723800003.0001") -> dict:
    """Somebody typing in the thread. No ``bot_id``, so the grammar reads it."""
    return {"ts": ts, "user": ACTOR, "text": text}


def _signal_reply(reactions=None) -> dict:
    """One of *our* signal replies as Slack returns it, reactions included.

    ``bot_id`` because we posted it: `is_human_message` must never read one of
    these back as somebody typing.
    """
    return {
        "ts": "1723800001.0002",
        "bot_id": "B1",
        "text": "Ada · WIP sprawl — 👍 if that's right, 👎 if it isn't",
        "reactions": reactions if reactions is not None else [{"name": "+1", "count": 1, "users": [ACTOR]}],
    }


class TestDeclines:
    """Every one of these writes a row saying which decline it was."""

    def test_no_token_never_calls_slack(self, db, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "")
        api = FakeApi()
        result = run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert result.outcome == POLL_NO_TOKEN
        assert result.declined

    def test_no_channel_declines(self, db, monkeypatch):
        monkeypatch.setenv("SLACK_CHANNEL_ID", "")
        result = run_poll(db_path=db, now=NOW, api=FakeApi(), apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert result.outcome == POLL_NO_CHANNEL

    def test_an_empty_allowlist_declines_without_calling_slack(self, db, monkeypatch):
        # With nobody authorised there is no event this poll could act on, so
        # the API call would be pure waste.
        monkeypatch.setenv("SLACK_ALLOWED_MEMBER_IDS", "")
        called = []

        class _Counting(FakeApi):
            def auth_test(self, *, token="", budget=None):
                called.append("auth")
                return super().auth_test()

        result = run_poll(db_path=db, now=NOW, api=_Counting(), apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert result.outcome == POLL_NO_ALLOWLIST
        assert called == []

    def test_a_malformed_allowlist_is_the_same_as_an_empty_one(self, db, monkeypatch):
        # A half-filled list LOOKS configured, which is why one bad entry voids
        # all of it rather than quietly dropping itself.
        monkeypatch.setenv("SLACK_ALLOWED_MEMBER_IDS", f"{ACTOR},oops")
        result = run_poll(db_path=db, now=NOW, api=FakeApi(), apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert result.outcome == POLL_NO_ALLOWLIST

    def test_a_declined_poll_is_still_recorded(self, db, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "")
        run_poll(db_path=db, now=NOW, api=FakeApi(), apply_event=lambda e, **_kw: ApplyResult(True, ""))
        with SlackStore(db) as store:
            assert store.polls()[0]["outcome"] == POLL_NO_TOKEN

    def test_a_second_poll_backs_off_rather_than_racing(self, db, monkeypatch):
        # launchd will not double-run a loaded label; cron absolutely will.
        monkeypatch.setattr(poller, "_lock", lambda path: None)
        result = run_poll(db_path=db, now=NOW, api=FakeApi(), apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert result.outcome == POLL_LOCKED


class TestApplying:
    def test_an_allowlisted_reaction_is_applied_once(self, db):
        _anchor(db)
        seen = []
        result = run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(reactions=_react()),
            apply_event=lambda e, **_kw: (seen.append(e.intent), ApplyResult(True, "paused"))[1],
        )
        assert result.outcome == POLL_OK
        assert result.events_applied == 1
        assert seen == ["pause"]

    def test_replaying_the_same_window_applies_nothing_further(self, db):
        # The window overlaps ~288x at a ten-minute cadence. That is only free
        # because the ledger's key makes the second sighting a no-op.
        _anchor(db)
        api = FakeApi(reactions=_react())
        applied = []
        for _ in range(3):
            run_poll(
                db_path=db,
                now=NOW,
                api=api,
                apply_event=lambda e, **_kw: (applied.append(e), ApplyResult(True, ""))[1],
            )
        assert len(applied) == 1

    def test_an_unauthorised_actor_is_ignored_silently_but_recorded(self, db):
        _anchor(db)
        applied = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(reactions=_react(users=["U5555555555"])),
            apply_event=lambda e, **_kw: (applied.append(e), ApplyResult(True, ""))[1],
        )
        assert applied == []
        with SlackStore(db) as store:
            assert store.history()[0]["outcome"] == "unauthorized"

    def test_the_bot_may_not_authorise_itself(self, db, monkeypatch):
        # Otherwise the acknowledgement reaction drives the next round.
        monkeypatch.setenv("SLACK_ALLOWED_MEMBER_IDS", "UBOT000000")
        _anchor(db)
        applied = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(reactions=_react(users=["UBOT000000"])),
            apply_event=lambda e, **_kw: (applied.append(e), ApplyResult(True, ""))[1],
        )
        assert applied == []

    def test_an_emoji_outside_the_grammar_is_recorded_as_ignored(self, db):
        _anchor(db)
        applied = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(reactions=_react(emoji="tada")),
            apply_event=lambda e, **_kw: (applied.append(e), ApplyResult(True, ""))[1],
        )
        assert applied == []
        with SlackStore(db) as store:
            assert store.history()[0]["outcome"] == "ignored"

    def test_an_expired_anchor_resolves_to_nothing(self, db):
        # apply_verdict would refuse a stale run on its own, but a stale
        # ceremony control would happily fire.
        _anchor(db, expires_at=(NOW - timedelta(days=1)).isoformat(timespec="seconds"))
        applied = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(reactions=_react()),
            apply_event=lambda e, **_kw: (applied.append(e), ApplyResult(True, ""))[1],
        )
        assert applied == []
        with SlackStore(db) as store:
            assert store.history()[0]["outcome"] == "stale"

    def test_a_thumb_on_the_post_is_not_a_verdict(self, db):
        _anchor(db)
        applied = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(reactions=_react(emoji="+1")),
            apply_event=lambda e, **_kw: (applied.append(e), ApplyResult(True, ""))[1],
        )
        assert applied == []

    def test_a_thumb_on_a_signal_is_a_verdict_carrying_its_member_and_rule(self, db):
        # A signal is reached through the thread of the post it hangs under,
        # never iterated on its own: that would spend one `reactions.get` per
        # signal where the thread read the poll already makes has them all.
        _anchor(db)
        _anchor(
            db, kind=KIND_SIGNAL, ts="1723800001.0002", root_ts="1723800000.000100", member="Ada", rule="wip-sprawl"
        )
        seen = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(thread={"1723800000.000100": [_signal_reply()]}),
            apply_event=lambda e, **_kw: (
                seen.append((e.act, e.intent, e.anchor.member, e.anchor.rule)),
                ApplyResult(True),
            )[1],
        )
        assert seen == [("verdict", "up", "Ada", "wip-sprawl")]

    def test_a_whole_reaction_list_costs_no_extra_api_call(self, db):
        _anchor(db)
        _anchor(
            db, kind=KIND_SIGNAL, ts="1723800001.0002", root_ts="1723800000.000100", member="Ada", rule="wip-sprawl"
        )
        reads = []

        class _Counting(FakeApi):
            def reactions_get(self, channel, ts, *, token="", budget=None):
                reads.append(ts)
                return super().reactions_get(channel, ts, token=token)

        run_poll(
            db_path=db,
            now=NOW,
            api=_Counting(thread={"1723800000.000100": [_signal_reply()]}),
            apply_event=lambda e, **_kw: ApplyResult(True),
        )
        # Only the post's. The signal's came out of the thread read.
        assert reads == ["1723800000.000100"]

    def test_a_truncated_reaction_list_is_detected_and_re_read_in_full(self, db):
        # `conversations.replies` caps `users` at ~25, and a truncated list is
        # indistinguishable from a short one — except that `count` disagrees
        # with it. A vote dropped here would be dropped with nothing to notice.
        _anchor(db)
        _anchor(
            db, kind=KIND_SIGNAL, ts="1723800001.0002", root_ts="1723800000.000100", member="Ada", rule="wip-sprawl"
        )
        reads = []

        class _Counting(FakeApi):
            def reactions_get(self, channel, ts, *, token="", budget=None):
                reads.append(ts)
                if ts == "1723800001.0002":
                    return SlackResponse(ok=True, data={"message": {"reactions": [{"name": "+1", "users": [ACTOR]}]}})
                return super().reactions_get(channel, ts, token=token)

        seen = []
        reply = _signal_reply(reactions=[{"name": "+1", "count": 30, "users": ["UZZZZZZZZZ"]}])
        run_poll(
            db_path=db,
            now=NOW,
            api=_Counting(thread={"1723800000.000100": [reply]}),
            apply_event=lambda e, **_kw: (seen.append(e.slack_user), ApplyResult(True))[1],
        )
        assert "1723800001.0002" in reads
        assert seen == [ACTOR]  # the allowlisted voter the inline list had hidden

    def test_a_refusal_is_recorded_with_its_reason(self, db):
        _anchor(db)
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(reactions=_react()),
            apply_event=lambda e, **_kw: ApplyResult(False, "already paused"),
        )
        with SlackStore(db) as store:
            row = store.history()[0]
        assert (row["outcome"], row["reason"]) == ("refused", "already paused")

    def test_one_raising_event_does_not_stop_the_others(self, db):
        _anchor(db)
        _anchor(db, ts="1723800009.0009")
        calls = []

        def _apply(event, **_kw):
            calls.append(event.anchor_ts)
            if event.anchor_ts.endswith("0009"):
                raise RuntimeError("boom")
            return ApplyResult(True, "ok")

        result = run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(
                reactions={
                    "1723800000.000100": [{"name": "pause_button", "users": [ACTOR]}],
                    "1723800009.0009": [{"name": "pause_button", "users": [ACTOR]}],
                }
            ),
            apply_event=_apply,
        )
        assert result.outcome == POLL_OK
        assert len(calls) == 2
        with SlackStore(db) as store:
            assert {r["outcome"] for r in store.history()} == {"applied", "failed"}


class TestReplies:
    def test_a_bare_verb_in_a_thread_acts(self, db):
        _anchor(db)
        seen = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(thread={"1723800000.000100": [{"ts": "1723800005.5", "user": ACTOR, "text": "pause"}]}),
            apply_event=lambda e, **_kw: (seen.append(e.intent), ApplyResult(True, ""))[1],
        )
        assert seen == ["pause"]

    def test_a_bot_reply_is_never_read(self, db):
        # Another integration talking in the thread must not drive yeaboi.
        _anchor(db)
        applied = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(
                thread={"1723800000.000100": [{"ts": "1723800005.5", "bot_id": "B1", "user": ACTOR, "text": "pause"}]}
            ),
            apply_event=lambda e, **_kw: (applied.append(e), ApplyResult(True, ""))[1],
        )
        assert applied == []

    def test_an_edited_message_shape_is_not_read(self, db):
        _anchor(db)
        applied = []
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(
                thread={
                    "1723800000.000100": [
                        {"ts": "1723800005.5", "user": ACTOR, "text": "pause", "subtype": "message_changed"}
                    ]
                }
            ),
            apply_event=lambda e, **_kw: (applied.append(e), ApplyResult(True, ""))[1],
        )
        assert applied == []


class TestIsHumanMessage:
    @pytest.mark.parametrize(
        "message",
        [
            {"user": "U1", "text": "hi"},
            {"user": "U1", "text": "hi", "subtype": ""},
            {"user": "U1", "text": "hi", "subtype": "thread_broadcast"},
        ],
    )
    def test_a_plain_message_is_human(self, message):
        assert is_human_message(message)

    @pytest.mark.parametrize(
        "message",
        [
            {"bot_id": "B1", "user": "U1"},
            {"app_id": "A1", "user": "U1"},
            {"user": "", "text": "hi"},
            {"text": "hi"},
            # username is client-settable, so it is never a fallback for user.
            {"username": "ada", "text": "hi"},
            {"user": "U1", "subtype": "message_deleted"},
            "not a dict",
        ],
    )
    def test_everything_else_is_not(self, message):
        assert not is_human_message(message)


class TestAck:
    def test_off_by_default(self, db):
        # reactions:write is the scope an administrator is most likely to
        # refuse, and this is a read feature.
        _anchor(db)
        api = FakeApi(reactions=_react())
        run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert api.acks == []

    def test_ticks_the_message_when_configured(self, db, monkeypatch):
        monkeypatch.setenv("SLACK_ACK_REACTION", "white_check_mark")
        _anchor(db)
        api = FakeApi(reactions=_react())
        run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert api.acks == [("C123", "1723800000.000100", "white_check_mark")]

    def test_never_ticks_something_it_did_not_apply(self, db, monkeypatch):
        monkeypatch.setenv("SLACK_ACK_REACTION", "white_check_mark")
        _anchor(db)
        api = FakeApi(reactions=_react())
        run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(False, "nope"))
        assert api.acks == []


class TestFailures:
    def test_a_revoked_token_fails_the_poll_with_slacks_own_code(self, db):
        api = FakeApi(auth=SlackResponse(ok=False, error="token_revoked"))
        result = run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert (result.outcome, result.error) == ("failed", "token_revoked")

    def test_an_unreadable_message_does_not_lose_the_rest_of_the_window(self, db):
        _anchor(db)
        _anchor(db, ts="1723800009.0009")
        api = FakeApi(
            reactions={
                "1723800000.000100": SlackResponse(ok=False, error="message_not_found"),
                "1723800009.0009": [{"name": "pause_button", "users": [ACTOR]}],
            }
        )
        result = run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(True, "ok"))
        assert result.events_applied == 1

    def test_a_gap_longer_than_the_window_is_reported_not_widened(self, db):
        # Acting on a three-day-old approval is the staleness the ceremonies
        # engine already ruled against; doing it silently is worse.
        with SlackStore(db) as store:
            store.record_poll({"polled_at": (NOW - timedelta(days=4)).isoformat(timespec="seconds"), "outcome": "ok"})
        result = run_poll(db_path=db, now=NOW, api=FakeApi(), apply_event=lambda e, **_kw: ApplyResult(True, ""))
        assert "exceeds the 48h window" in result.detail


class TestPollResult:
    @pytest.mark.parametrize(
        "outcome,declined",
        [(POLL_OK, False), ("failed", False), (POLL_NO_TOKEN, True), (POLL_LOCKED, True)],
    )
    def test_declined_separates_a_refusal_from_a_failure(self, outcome, declined):
        assert PollResult(outcome=outcome).declined is declined


class TestSpeaking:
    """A ✅ and a thread line answer two different questions."""

    def test_a_deferral_gets_both_a_tick_and_a_line(self, db, monkeypatch):
        monkeypatch.setenv("SLACK_ACK_REACTION", "white_check_mark")
        _anchor(db)
        api = FakeApi(reactions=_react())
        run_poll(
            db_path=db,
            now=NOW,
            api=api,
            apply_event=lambda e, **_kw: ApplyResult(
                True, "recorded — someone is correcting this report", "deferred", True
            ),
        )
        assert api.acks and api.said
        assert "recorded" in api.said[0][1]

    def test_a_refusal_that_asked_for_silence_gets_it(self, db):
        # A prose reply under a post with nothing correctable behind it refuses
        # here, and that is true of *every* sentence in that thread. A bot that
        # answered all of them is one nobody leaves switched on.
        _anchor(db)
        api = FakeApi(reactions=_react())
        run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(False, "nothing correctable"))
        assert api.said == []

    def test_an_unauthorised_actor_is_never_answered(self, db):
        # The channel is not the place to announce who is unauthorised, and a
        # bot that answers unknown users is one anybody can make spam it.
        _anchor(db)
        api = FakeApi(reactions=_react(users=["U5555555555"]))
        run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(True, "", speak=True))
        assert api.said == []
        assert api.acks == []

    def test_a_line_about_a_signal_lands_in_the_post_s_thread(self, db):
        # Not under the signal reply: Slack threads are flat, so a `thread_ts`
        # of anything but the root would either be rejected or start a second
        # thread nobody is reading.
        _anchor(db)
        _anchor(
            db, kind=KIND_SIGNAL, ts="1723800001.0002", root_ts="1723800000.000100", member="Ada", rule="wip-sprawl"
        )
        api = FakeApi(thread={"1723800000.000100": [_signal_reply()]})
        run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(True, "noted", speak=True))
        assert api.said == [("1723800000.000100", "noted")]


class TestCorrectionsReachApply:
    """The whole of what a typed sentence carries by the time it is applied."""

    def _seen(self, db, text):
        seen: list = []
        _anchor(db, artifact_kind="standup")
        run_poll(
            db_path=db,
            now=NOW,
            api=FakeApi(thread={"1723800000.000100": [_human_reply(text)]}),
            apply_event=lambda e, **_kw: (seen.append(e), ApplyResult(True, "noted", speak=True))[1],
        )
        return seen

    def test_the_text_and_the_reply_it_came_from_both_arrive(self, db):
        # `reply_ts` is the deterministic edit id, so a correction with no reply
        # behind it has no idempotency key at all.
        (event,) = self._seen(db, "Ada was on leave, that ticket is not hers")
        assert event.act == "correction"
        assert event.payload.startswith("Ada was on leave")
        assert event.reply_ts == "1723800003.0001"

    def test_the_text_is_already_clean_when_it_arrives(self, db):
        # `clean_reply_text` runs in the grammar, upstream of everything, so
        # nothing that pings a workspace can reach the store even in principle.
        (event,) = self._seen(db, "<!channel> Ada was on leave <@U0999>")
        assert "<!" not in event.payload and "<@" not in event.payload

    def test_an_acknowledgement_never_becomes_an_event_to_apply(self, db):
        assert self._seen(db, "ok") == []

    def test_a_successful_correction_answers_in_the_thread(self, db):
        # The ✅ needs `reactions:write` and is off by default, so without this
        # a write of somebody's own prose into a stored report lands with no
        # signal whatsoever that it worked.
        _anchor(db, artifact_kind="standup")
        api = FakeApi(thread={"1723800000.000100": [_human_reply()]})
        run_poll(db_path=db, now=NOW, api=api, apply_event=lambda e, **_kw: ApplyResult(True, "noted", speak=True))
        assert api.said == [("1723800000.000100", "noted")]


class TestOnePollReadsAndWritesOneDatabase:
    """``db_path`` reaches the applier, not just the poller's own stores.

    It used to stop at ``_handle``: the poll read anchors and wrote its ledger
    into the database it was handed, while the applier fell back to
    ``get_db_path()`` and paused a ceremony, cast a verdict and appended a note
    in the operator's real one. The claim row saying the event was handled
    landed on the wrong side, so replaying the same window against the real
    database would apply it a second time.

    Nothing shipping hit it — both callers leave ``db_path`` at None — and
    nothing could see it either, because every stub here took ``event`` alone.
    """

    def test_the_applier_is_handed_the_polls_own_database(self, db):
        seen: list = []

        def _apply(event, *, db_path=None):
            seen.append(db_path)
            return ApplyResult(True, "")

        _anchor(db)
        run_poll(db_path=db, now=NOW, api=FakeApi(reactions=_react()), apply_event=_apply)
        assert seen == [db], "the applier writes where the poll reads, or the claim row lies"
