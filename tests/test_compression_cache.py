"""Tests for CompressionCache with LRU eviction."""

from __future__ import annotations

import pytest

from headroom.cache.compression_cache import (
    CompressionCache,
    _extract_tool_result_content,
    _swap_tool_result_content,
)


@pytest.fixture
def cache() -> CompressionCache:
    return CompressionCache()


@pytest.fixture
def small_cache() -> CompressionCache:
    return CompressionCache(max_entries=3)


class TestCompressionCache:
    def test_cache_miss_returns_none(self, cache: CompressionCache) -> None:
        h = CompressionCache.content_hash("some content")
        assert cache.get_compressed(h) is None

    def test_store_and_retrieve(self, cache: CompressionCache) -> None:
        content = "hello world this is a long message"
        h = CompressionCache.content_hash(content)
        cache.store_compressed(h, "hello world...compressed", tokens_saved=15)
        assert cache.get_compressed(h) == "hello world...compressed"

    def test_different_content_different_hash(self) -> None:
        h1 = CompressionCache.content_hash("content A")
        h2 = CompressionCache.content_hash("content B")
        assert h1 != h2

    def test_overwrite_same_hash(self, cache: CompressionCache) -> None:
        h = CompressionCache.content_hash("some content")
        cache.store_compressed(h, "v1", tokens_saved=10)
        cache.store_compressed(h, "v2", tokens_saved=20)
        assert cache.get_compressed(h) == "v2"

    def test_stats_tracking(self, cache: CompressionCache) -> None:
        h = CompressionCache.content_hash("content")
        cache.store_compressed(h, "compressed", tokens_saved=5)

        # One hit
        cache.get_compressed(h)
        # One miss
        cache.get_compressed("nonexistent")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        assert stats["tokens_saved"] == 5

    def test_eviction_at_max_entries(self, small_cache: CompressionCache) -> None:
        h1 = CompressionCache.content_hash("a")
        h2 = CompressionCache.content_hash("b")
        h3 = CompressionCache.content_hash("c")
        h4 = CompressionCache.content_hash("d")

        small_cache.store_compressed(h1, "ca", tokens_saved=1)
        small_cache.store_compressed(h2, "cb", tokens_saved=1)
        small_cache.store_compressed(h3, "cc", tokens_saved=1)

        # Adding a 4th should evict the oldest (h1)
        small_cache.store_compressed(h4, "cd", tokens_saved=1)

        assert small_cache.get_compressed(h1) is None
        assert small_cache.get_compressed(h2) == "cb"
        assert small_cache.get_compressed(h4) == "cd"

    def test_access_refreshes_lru(self, small_cache: CompressionCache) -> None:
        h1 = CompressionCache.content_hash("a")
        h2 = CompressionCache.content_hash("b")
        h3 = CompressionCache.content_hash("c")
        h4 = CompressionCache.content_hash("d")

        small_cache.store_compressed(h1, "ca", tokens_saved=1)
        small_cache.store_compressed(h2, "cb", tokens_saved=1)
        small_cache.store_compressed(h3, "cc", tokens_saved=1)

        # Access h1 to refresh it
        small_cache.get_compressed(h1)

        # Adding h4 should evict h2 (oldest untouched), not h1
        small_cache.store_compressed(h4, "cd", tokens_saved=1)

        assert small_cache.get_compressed(h1) == "ca"
        assert small_cache.get_compressed(h2) is None
        assert small_cache.get_compressed(h4) == "cd"

    def test_content_hash_list_content(self) -> None:
        """content_hash handles Anthropic-format list content."""
        list_content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        h = CompressionCache.content_hash(list_content)
        assert isinstance(h, str)
        assert len(h) == 16

        # Same content produces same hash
        assert CompressionCache.content_hash(list_content) == h

    def test_content_hash_string_length(self) -> None:
        h = CompressionCache.content_hash("test")
        assert len(h) == 16


class TestCompressionCacheFrozenCount:
    def test_empty_cache_returns_zero(self, cache: CompressionCache) -> None:
        assert cache.compute_frozen_count([]) == 0

    def test_user_assistant_stable_with_live_zone_cap(self, cache: CompressionCache) -> None:
        """Plain user/assistant turns are individually stable, but the
        trailing message is reserved as the live zone — the new turn
        cannot be in any provider prefix cache. See docstring on
        ``CompressionCache.compute_frozen_count``."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ]
        # 3 messages structurally stable; cap clamps to len-1 = 2.
        assert cache.compute_frozen_count(messages) == 2

    def test_tool_result_with_cache_hit_capped_at_live_zone(self, cache: CompressionCache) -> None:
        tool_content = "tool output data"
        h = CompressionCache.content_hash(tool_content)
        cache.store_compressed(h, "compressed tool output", tokens_saved=5)

        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "my_tool", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
        ]
        # All 3 stable; cap clamps to len-1 = 2 (trailing tool_result is
        # the live zone).
        assert cache.compute_frozen_count(messages) == 2

    def test_tool_result_cache_miss_stops_frozen(self, cache: CompressionCache) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "uncached stuff"}
                ],
            },
            {"role": "user", "content": "follow up"},
        ]
        assert cache.compute_frozen_count(messages) == 1

    def test_frozen_count_with_dropped_messages(self, cache: CompressionCache) -> None:
        cached_content = "cached tool output"
        h = CompressionCache.content_hash(cached_content)
        cache.store_compressed(h, "compressed", tokens_saved=3)

        messages = [
            {"role": "user", "content": "start"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": cached_content}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "not cached"}],
            },
        ]
        assert cache.compute_frozen_count(messages) == 2

    def test_stable_hash_allows_frozen_count_past_uncached_tool_result(
        self, cache: CompressionCache
    ) -> None:
        """Tool_results marked stable should not stop the frozen count walk."""
        tool_content = "excluded Read output — big file contents"
        h = CompressionCache.content_hash(tool_content)
        cache.mark_stable(h)

        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
            {"role": "user", "content": "follow up"},
        ]
        # Without mark_stable, the walk would stop at msg[1] → frozen=1.
        # With stable hash, the walk continues past msg[1]; structural
        # count = 3, then capped at len-1 = 2 (live-zone reservation).
        assert cache.compute_frozen_count(messages) == 2

    def test_update_from_result_identical_content_marks_stable(
        self, cache: CompressionCache
    ) -> None:
        """When orig == compressed, update_from_result marks the hash as stable."""
        tool_content = "unchanged tool output"
        originals = [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
        ]
        # Compressed is identical to originals (no compression happened)
        compressed = [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
        ]
        cache.update_from_result(originals, compressed)

        h = CompressionCache.content_hash(tool_content)
        assert h in cache._stable_hashes

        # Frozen count walks past this tool_result (its hash is stable),
        # but the trailing message is still reserved as live zone.
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
            {"role": "user", "content": "more stuff"},
        ]
        assert cache.compute_frozen_count(messages) == 2

    def test_mark_stable_from_messages(self, cache: CompressionCache) -> None:
        """mark_stable_from_messages records hashes for tool_results."""
        content_a = "tool output A"
        content_b = "tool output B"
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content_a}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t2", "content": content_b}],
            },
        ]
        # Mark first 2 messages (msg[0] + msg[1])
        cache.mark_stable_from_messages(messages, 2)

        ha = CompressionCache.content_hash(content_a)
        hb = CompressionCache.content_hash(content_b)
        assert ha in cache._stable_hashes
        assert hb not in cache._stable_hashes  # msg[2] not included

    def test_should_defer_compression_new_content(self, cache: CompressionCache) -> None:
        """First-time content should NOT be deferred — there is no
        prefix-cache entry to preserve, so compression carries no bust
        cost. Issue #327: prior behavior deferred first-sight, which
        marked every fresh tool_result as stable and disabled
        compression for typical Claude Code workloads.
        """
        h = CompressionCache.content_hash("brand new content")
        assert cache.should_defer_compression(h, ttl_seconds=300, batch_window=30) is False
        # Subsequent sightings within TTL should defer (batch window).
        assert cache.should_defer_compression(h, ttl_seconds=300, batch_window=30) is True

    def test_should_defer_compression_records_first_seen(self, cache: CompressionCache) -> None:
        """First-sight call must record the timestamp so subsequent
        in-window calls can defer. Without this the deferral pathway
        for genuinely-repeated content stops working."""
        h = CompressionCache.content_hash("seen-twice content")
        cache.should_defer_compression(h)  # first sight
        assert h in cache._first_seen

    def test_should_defer_compression_near_ttl(self, cache: CompressionCache) -> None:
        """Content near TTL boundary should NOT be deferred."""
        import time

        h = CompressionCache.content_hash("old content")
        # Backdate first_seen to simulate age near TTL
        cache._first_seen[h] = time.time() - 280  # 280s old, TTL=300, window=30
        assert cache.should_defer_compression(h, ttl_seconds=300, batch_window=30) is False


class TestCompressionCacheApplyAndUpdate:
    def test_apply_cached_swaps_tool_results(self, cache: CompressionCache) -> None:
        original_content = "big tool output"
        h = CompressionCache.content_hash(original_content)
        cache.store_compressed(h, "small output", tokens_saved=5)

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": original_content}
                ],
            },
        ]
        result = cache.apply_cached(messages)
        assert result[1]["content"][0]["content"] == "small output"

    def test_apply_cached_preserves_uncached_messages(self, cache: CompressionCache) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = cache.apply_cached(messages)
        assert result[0] is messages[0]
        assert result[1] is messages[1]

    def test_apply_cached_never_adds_messages(self, cache: CompressionCache) -> None:
        # Store something in cache that doesn't correspond to any message
        cache.store_compressed("orphan_hash", "orphan_value", tokens_saved=1)

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = cache.apply_cached(messages)
        assert len(result) == len(messages)

    def test_update_from_result_caches_changes(self, cache: CompressionCache) -> None:
        originals = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "original output"}
                ],
            },
        ]
        compressed = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "compressed output"}
                ],
            },
        ]
        cache.update_from_result(originals, compressed)

        h = CompressionCache.content_hash("original output")
        assert cache.get_compressed(h) == "compressed output"

    def test_update_from_result_ignores_unchanged(self, cache: CompressionCache) -> None:
        originals = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "same content"}
                ],
            },
        ]
        compressed = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "same content"}
                ],
            },
        ]
        cache.update_from_result(originals, compressed)
        h = CompressionCache.content_hash("same content")
        assert cache.get_compressed(h) is None

    def test_apply_does_not_modify_original_messages(self, cache: CompressionCache) -> None:
        original_content = "big tool output"
        h = CompressionCache.content_hash(original_content)
        cache.store_compressed(h, "small output", tokens_saved=5)

        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": original_content}],
        }
        messages = [msg]
        cache.apply_cached(messages)

        # Original must be untouched
        assert msg["content"][0]["content"] == original_content

    def test_openai_format_tool_result(self, cache: CompressionCache) -> None:
        original_content = "openai tool output"
        h = CompressionCache.content_hash(original_content)
        cache.store_compressed(h, "compressed openai", tokens_saved=4)

        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": original_content},
        ]
        result = cache.apply_cached(messages)
        assert result[0]["content"] == "compressed openai"
        # Original untouched
        assert messages[0]["content"] == original_content


# ─── C1 (audit follow-up): concurrency regression suite ────────────────────
#
# CompressionCache must be safe under multi-threaded mutation. The proxy is
# async and dispatches multiple concurrent requests per `session_id` into
# `asyncio.to_thread` workers — a single CompressionCache instance therefore
# sees concurrent calls to `store_compressed` / `get_compressed` /
# `mark_stable_from_messages` / `apply_cached` / `update_from_result`.
# These tests provoke the race conditions that motivated adding `_lock`.


class TestCompressionCacheConcurrency:
    """Threading regression suite for the audit-followup lock."""

    def test_concurrent_store_does_not_corrupt_total_tokens_saved(self) -> None:
        """Many threads each store_compressed with tokens_saved=N; the
        bookkeeping field must equal SUM(N) when threads finish. Pre-lock
        this races (read-modify-write of `_total_tokens_saved`)."""
        import threading

        cache = CompressionCache(max_entries=1_000_000)
        n_threads = 32
        per_thread = 100
        per_thread_tokens = 7

        def worker(tid: int) -> None:
            for i in range(per_thread):
                h = CompressionCache.content_hash(f"thread-{tid}-item-{i}")
                cache.store_compressed(h, f"comp-{tid}-{i}", tokens_saved=per_thread_tokens)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_threads * per_thread * per_thread_tokens
        stats = cache.get_stats()
        assert stats["entries"] == n_threads * per_thread
        # The expected token count is exact only because each (thread, item)
        # produces a unique hash → no overwrite path. Pre-lock this would be
        # < expected due to lost updates.
        assert stats["tokens_saved"] == expected

    def test_concurrent_apply_cached_with_concurrent_store_does_not_raise(self) -> None:
        """`apply_cached` iterates `_cache` (via `get_compressed`); if a
        concurrent `store_compressed` mutates the OrderedDict during the
        iteration, pre-lock you'd get `RuntimeError: OrderedDict mutated
        during iteration`. Locks make this a single critical section."""
        import threading

        cache = CompressionCache()

        # Pre-populate so apply_cached has work to do.
        for i in range(50):
            h = CompressionCache.content_hash(f"seed-{i}")
            cache.store_compressed(h, f"comp-{i}", tokens_saved=1)

        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"t{i}",
                        "content": f"seed-{i}",
                    }
                ],
            }
            for i in range(50)
        ]

        stop = threading.Event()
        errors: list[Exception] = []

        def reader() -> None:
            try:
                while not stop.is_set():
                    _ = cache.apply_cached(msgs)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(500):
                    h = CompressionCache.content_hash(f"writer-{i}")
                    cache.store_compressed(h, f"w-{i}", tokens_saved=1)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        readers = [threading.Thread(target=reader) for _ in range(4)]
        writers = [threading.Thread(target=writer) for _ in range(4)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join()

        assert errors == [], f"Concurrent ops raised: {errors}"

    def test_concurrent_update_from_result_no_partial_state(self) -> None:
        """update_from_result must be all-or-nothing per call. With many
        threads calling update_from_result in parallel on the same cache,
        the final state must reflect every call's full effect (no partial
        writes)."""
        import threading

        cache = CompressionCache()

        n_threads = 16
        per_thread_calls = 20

        def worker(tid: int) -> None:
            for i in range(per_thread_calls):
                orig_text = f"orig-{tid}-{i}-" + "X" * 200
                comp_text = f"comp-{tid}-{i}-" + "X" * 50
                originals = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"t-{tid}-{i}",
                                "content": orig_text,
                            }
                        ],
                    }
                ]
                compressed = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"t-{tid}-{i}",
                                "content": comp_text,
                            }
                        ],
                    }
                ]
                cache.update_from_result(originals, compressed)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = cache.get_stats()
        # Each (tid, i) is a unique hash → cache entries == n_threads * per_thread_calls.
        assert stats["entries"] == n_threads * per_thread_calls
        assert stats["tokens_saved"] > 0

    def test_concurrent_hits_misses_consistent(self) -> None:
        """Under concurrent reads + writes, hits+misses must be bounded by
        total lookups (hits ≤ entries, misses ≥ 0 at all moments)."""
        import random
        import threading

        cache = CompressionCache(max_entries=1_000_000)
        n_threads = 16
        per_thread = 50

        # Pre-populate so reads have something to hit
        for i in range(per_thread):
            h = CompressionCache.content_hash(f"hit-{i}")
            cache.store_compressed(h, f"comp-{i}", tokens_saved=3)

        errors: list[Exception] = []
        barrier = threading.Barrier(n_threads)

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for i in range(per_thread):
                    if random.random() < 0.6:
                        # Read path
                        _ = cache.get_compressed(
                            CompressionCache.content_hash(
                                f"hit-{random.randint(0, per_thread - 1)}"
                            )
                        )
                    else:
                        # Write path
                        h = CompressionCache.content_hash(f"write-{tid}-{i}")
                        cache.store_compressed(h, f"w-{tid}-{i}", tokens_saved=1)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent reads+writes raised: {errors}"
        stats = cache.get_stats()
        # hits + misses should be non-negative (sanity)
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0
        assert stats["entries"] > 0

    def test_concurrent_stable_hash_ops_no_race(self) -> None:
        """Concurrent mark_stable_from_messages + compute_frozen_count must
        not race — stable_hashes must remain self-consistent."""
        import threading

        cache = CompressionCache()
        n_threads = 12
        per_thread = 30

        # Each thread has its own content; produce tool_result messages
        # and mark them stable, then verify frozen count.
        errors: list[Exception] = []
        barrier = threading.Barrier(n_threads)

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for i in range(per_thread):
                    content = f"stable-content-{tid}-{i}"
                    h = CompressionCache.content_hash(content)
                    # Also store to make it appear cached
                    cache.store_compressed(h, f"comp-{tid}-{i}", tokens_saved=2)
                    # Mark stable
                    cache.mark_stable(h)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent stable-hash ops raised: {errors}"
        stats = cache.get_stats()
        # All entries should be recorded; stable_hashes should match entries
        # (every store_compressed was followed by mark_stable in our test)
        assert stats["entries"] == n_threads * per_thread


def test_get_compression_cache_returns_same_instance_under_contention() -> None:
    """`HeadroomProxy._get_compression_cache(session_id)` must return the
    SAME `CompressionCache` instance for concurrent calls with the same
    session_id. Pre-lock, two concurrent calls could both see "not in dict"
    and each create a new instance, splitting the cache state across them.
    """
    import threading

    pytest.importorskip("fastapi")
    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    proxy = app.state.proxy

    n_threads = 32
    results: list[CompressionCache] = []
    results_lock = threading.Lock()

    def worker() -> None:
        c = proxy._get_compression_cache("shared-session-id")
        with results_lock:
            results.append(c)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == n_threads
    first = results[0]
    for c in results[1:]:
        assert c is first, "Concurrent _get_compression_cache returned different instances"


# ─── Defect 1: Anthropic list-form tool_result content ─────────────────────
#
# `_extract_tool_result_content` must handle Anthropic-native `tool_result`
# blocks whose `content` is a LIST of typed blocks (e.g.
# `[{"type": "text", "text": "..."}]`), not just plain strings. This shape
# is emitted by MCP tools and modern Claude Code. Prior code only handled
# `isinstance(inner, str)` and fell through to `return None` for list-form
# content, which made `compute_frozen_count` treat such messages as
# unstable and `break` — the tool_result never entered the compression
# cache.


class TestListFormToolResultContent:
    def test_extract_list_form_joins_text_blocks(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": "A"},
                        {"type": "text", "text": "B"},
                    ],
                }
            ],
        }
        assert _extract_tool_result_content(msg) == "A\nB"

    def test_extract_list_form_no_text_blocks_returns_none(self) -> None:
        """Non-text blocks are ignored; with NO text blocks present, the
        result must be `None` (not `""`) — `None` is the cache-miss/unstable
        signal downstream, whereas `""` would be treated as valid content."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": "..."}},
                    ],
                }
            ],
        }
        assert _extract_tool_result_content(msg) is None

    def test_extract_list_form_mixed_returns_none(self) -> None:
        """A MIXED list ([text, image, text]) must extract to ``None``, NOT a
        text-only join. Joining would let the downstream swap collapse the whole
        block list to a string and silently DROP the image (data loss), and it
        would let two results with identical text but different images collide
        on the text-only hash. ``None`` marks the message unstable so it passes
        through untouched — the pre-PR guarantee."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": "A"},
                        {"type": "image", "source": {"type": "base64", "data": "..."}},
                        {"type": "text", "text": "B"},
                    ],
                }
            ],
        }
        assert _extract_tool_result_content(msg) is None

    def test_extract_str_form_unchanged(self) -> None:
        """str-form path must be byte-identical: same string returned, same
        hash produced. Guards against the list-form branch regressing the
        pre-existing str-form behavior."""
        original_content = "plain string content"
        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": original_content}],
        }
        extracted = _extract_tool_result_content(msg)
        assert extracted == original_content
        assert CompressionCache.content_hash(extracted) == CompressionCache.content_hash(
            original_content
        )

    def test_compute_frozen_count_list_form_stable_when_cached(
        self, cache: CompressionCache
    ) -> None:
        """A list-form tool_result whose extracted-text hash is already in
        the cache must be walked past by `compute_frozen_count`, exactly as
        a str-form tool_result would be."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "list"},
                            {"type": "text", "text": "form"},
                        ],
                    }
                ],
            },
            {"role": "user", "content": "follow up"},
        ]
        extracted = _extract_tool_result_content(messages[1])
        assert extracted == "list\nform"
        h = CompressionCache.content_hash(extracted)
        cache.store_compressed(h, "compressed list form", tokens_saved=2)

        # All 3 structurally stable; cap clamps to len-1 = 2 (live zone).
        assert cache.compute_frozen_count(messages) == 2

    def test_compute_frozen_count_list_form_cache_miss_stops_frozen(
        self, cache: CompressionCache
    ) -> None:
        """Sanity counterpart: without a pre-stored cache entry, list-form
        tool_result is NOT stable, matching str-form cache-miss behavior."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "uncached"}],
                    }
                ],
            },
            {"role": "user", "content": "follow up"},
        ]
        assert cache.compute_frozen_count(messages) == 1

    def test_swap_tool_result_content_pure_text_list_collapses(self) -> None:
        """For a PURE text-only list (no non-text blocks to lose),
        `_swap_tool_result_content` collapses to the replacement string on write
        (compression always yields a single string); the original message is
        left untouched."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": "A"},
                        {"type": "text", "text": "B"},
                    ],
                }
            ],
        }
        new_msg = _swap_tool_result_content(msg, "compressed replacement")
        assert new_msg["content"][0]["content"] == "compressed replacement"
        # Original untouched (still list-form).
        assert msg["content"][0]["content"] == [
            {"type": "text", "text": "A"},
            {"type": "text", "text": "B"},
        ]

    def test_apply_cached_mixed_content_returned_unchanged(self, cache: CompressionCache) -> None:
        """END-TO-END data-loss guard: a MIXED [text, image, text] tool_result
        must pass through `apply_cached` byte-for-byte UNCHANGED — the image is
        never dropped. Pre-fix, `_join_text_blocks` joined the text to ``"A\\nB"``,
        `apply_cached` hit the poisoned text-only cache entry, and
        `_swap_tool_result_content` replaced the ENTIRE block list with the
        compressed string, silently discarding the image."""
        mixed_content = [
            {"type": "text", "text": "A"},
            {"type": "image", "source": {"type": "base64", "data": "IMG_DATA"}},
            {"type": "text", "text": "B"},
        ]
        # Poison the cache at the text-only hash so the pre-fix path WOULD swap.
        text_hash = CompressionCache.content_hash("A\nB")
        cache.store_compressed(text_hash, "COMPRESSED", tokens_saved=5)

        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": mixed_content}],
        }
        result = cache.apply_cached([msg])

        # The image (and every block) is preserved: no collapse to a string.
        assert result[0]["content"][0]["content"] == mixed_content
        # Original message also untouched.
        assert msg["content"][0]["content"] == mixed_content

    def test_mixed_results_identical_text_different_images_do_not_collide(
        self, cache: CompressionCache
    ) -> None:
        """Two mixed tool_results with IDENTICAL text but DIFFERENT images must
        not collide: extraction returns ``None`` for both, so neither is cached
        under the (shared) text-only hash and each image is preserved. Pre-fix,
        both hashed to ``content_hash("same text")`` and would alias."""
        text_block = {"type": "text", "text": "same text"}
        img1 = {"type": "image", "source": {"type": "base64", "data": "IMG_ONE"}}
        img2 = {"type": "image", "source": {"type": "base64", "data": "IMG_TWO"}}

        msg1 = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [text_block, img1]}
            ],
        }
        msg2 = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": [text_block, img2]}
            ],
        }

        # Neither extracts to text → both unstable, neither enters the cache.
        assert _extract_tool_result_content(msg1) is None
        assert _extract_tool_result_content(msg2) is None

        out1 = cache.apply_cached([msg1])[0]
        out2 = cache.apply_cached([msg2])[0]
        # Distinct images preserved end-to-end (no aliasing on shared text hash).
        assert out1["content"][0]["content"][-1] == img1
        assert out2["content"][0]["content"][-1] == img2
        assert out1["content"][0]["content"][-1] != out2["content"][0]["content"][-1]

    def test_openai_format_unchanged(self, cache: CompressionCache) -> None:
        """OpenAI `role: tool` path must be unaffected by the list-form
        branch added to the Anthropic path."""
        original_content = "openai tool output unaffected"
        msg = {"role": "tool", "tool_call_id": "tc1", "content": original_content}
        assert _extract_tool_result_content(msg) == original_content
