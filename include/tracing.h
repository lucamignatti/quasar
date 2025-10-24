// tracing.h
//
// Lightweight Perfetto/Chrome-trace compatible tracer header.
// - Header-only, works on Linux/macOS
// - In-process tracing (single-run): call `tracing::Tracer::Start("trace.json")` at program start,
//   then `tracing::Tracer::Stop()` before exit.
// - Use `TRACE_SCOPE("name")` to instrument scopes (records "B"/"E" events).
// - Thread-safe: uses per-thread buffers that are periodically flushed to reduce memory.
//
// Usage example (in main):
//   tracing::Tracer::Get().Start("perfetto_trace.json");
//   TRACE_THREAD_NAME("main");
//   {
//     TRACE_SCOPE("program");
//     ... your code ...
//   }
//   tracing::Tracer::Get().Stop();
//
// Open the generated JSON in chrome://tracing or Perfetto UI (ui.perfetto.dev).

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tracing {

class Tracer {
public:
    struct Event {
        std::string name;
        char ph;            // 'B' = begin, 'E' = end, 'M' = metadata
        uint64_t ts;        // timestamp in microseconds since start
        uint64_t tid;       // thread id (hashed)
    };

    // Get singleton instance
    static Tracer& Get() {
        static Tracer instance;
        return instance;
    }

    // Start tracing. Must be called before any events are recorded.
    void Start(const std::string& filename = "trace.json") {
        std::lock_guard<std::mutex> lk(mutex_);
        if (started_.load()) return;
        outfile_.open(filename, std::ios::out | std::ios::trunc);
        if (!outfile_.is_open()) {
            std::cerr << "tracing: failed to open " << filename << " for writing\n";
            return;
        }
        outfile_ << "[";
        first_event_ = true;
        start_time_ = Clock::now();
        started_.store(true);
    }

    // Stop tracing and flush all thread buffers.
    void Stop() {
        if (!started_.exchange(false)) return;

        // Gather all thread buffers
        std::vector<std::vector<Event>*> buffers;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            for (auto& kv : thread_buffers_) {
                buffers.push_back(kv.second);
            }
        }

        // Collect and sort all events by timestamp
        std::vector<Event> all_events;
        for (auto* buf : buffers) {
            if (buf) {
                for (const auto& e : *buf) {
                    all_events.push_back(e);
                }
            }
        }

        std::sort(all_events.begin(), all_events.end(), [](const Event& a, const Event& b) {
            if (a.ts != b.ts) return a.ts < b.ts;
            if (a.tid != b.tid) return a.tid < b.tid;
            return a.name < b.name;
        });

        // Write all events
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (!outfile_.is_open()) return;

            for (const auto& e : all_events) {
                write_event(e);
            }

            outfile_ << "]\n";
            outfile_.close();
        }

        // Clean up thread buffers
        {
            std::lock_guard<std::mutex> lk(mutex_);
            for (auto& kv : thread_buffers_) {
                delete kv.second;
            }
            thread_buffers_.clear();
        }
    }

    // Record an event into the calling thread's buffer
    void Record(const std::string& name, char ph) {
        if (!started_.load(std::memory_order_acquire)) return;
        
        Event e;
        e.name = name;
        e.ph = ph;
        e.ts = timestamp_us();
        e.tid = thread_id_hash();
        
        get_thread_buffer()->push_back(std::move(e));
    }

    bool IsStarted() const { return started_.load(); }

private:
    Tracer() : started_(false), first_event_(true) {}
    ~Tracer() {
        if (started_.load()) {
            try { Stop(); } catch (...) {}
        }
    }

    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;

    using Clock = std::chrono::steady_clock;

    uint64_t timestamp_us() const {
        auto now = Clock::now();
        auto d = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
        return static_cast<uint64_t>(d.count());
    }

    static uint64_t thread_id_hash() {
        auto id = std::this_thread::get_id();
        return static_cast<uint64_t>(std::hash<std::thread::id>{}(id));
    }

    static std::string escape_json(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out += c; break;
            }
        }
        return out;
    }

    void write_event(const Event& e) {
        if (!first_event_) {
            outfile_ << ",";
        }
        first_event_ = false;

        outfile_ << "{";
        if (e.ph == 'M') {
            // Metadata event for thread name
            outfile_ << "\"name\":\"thread_name\",";
            outfile_ << "\"ph\":\"M\",";
            outfile_ << "\"ts\":" << e.ts << ",";
            outfile_ << "\"pid\":1,";
            outfile_ << "\"tid\":" << e.tid << ",";
            outfile_ << "\"args\":{\"name\":\"" << escape_json(e.name) << "\"}";
        } else {
            // Regular trace event
            outfile_ << "\"name\":\"" << escape_json(e.name) << "\",";
            outfile_ << "\"ph\":\"" << e.ph << "\",";
            outfile_ << "\"ts\":" << e.ts << ",";
            outfile_ << "\"pid\":1,";
            outfile_ << "\"tid\":" << e.tid;
        }
        outfile_ << "}";
    }

    std::vector<Event>* get_thread_buffer() {
        uint64_t tid = thread_id_hash();
        
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = thread_buffers_.find(tid);
        if (it == thread_buffers_.end()) {
            auto* buf = new std::vector<Event>();
            buf->reserve(4096);  // Pre-allocate to reduce allocations
            thread_buffers_[tid] = buf;
            return buf;
        }
        return it->second;
    }

    std::atomic<bool> started_;
    Clock::time_point start_time_;
    std::ofstream outfile_;
    bool first_event_;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::vector<Event>*> thread_buffers_;
};

// RAII scope tracer
class TraceScope {
public:
    explicit TraceScope(const std::string& name) : name_(name) {
        if (Tracer::Get().IsStarted()) {
            Tracer::Get().Record(name_, 'B');
        }
    }
    ~TraceScope() {
        if (Tracer::Get().IsStarted()) {
            Tracer::Get().Record(name_, 'E');
        }
    }
private:
    std::string name_;
};

// Macros
#define TRACE_SCOPE(name) ::tracing::TraceScope _trace_scope_##__LINE__ (name)
#define TRACE_THREAD_NAME(name) ::tracing::Tracer::Get().Record(name, 'M')

} // namespace tracing