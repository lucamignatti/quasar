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
        first_event_.store(true);
        start_time_ = Clock::now();
        started_.store(true);
    }

    // Stop tracing - just close the file (events already written during execution)
    void Stop() {
        if (!started_.exchange(false)) return;

        std::lock_guard<std::mutex> lk(mutex_);
        if (outfile_.is_open()) {
            outfile_ << "]\n";
            outfile_.close();
        }

        // Clean up thread buffers
        for (auto& kv : thread_buffers_) {
            delete kv.second;
        }
        thread_buffers_.clear();
    }

    // Record an event - write directly to file (streaming, no buffering)
    void Record(const std::string& name, char ph) {
        if (!started_.load(std::memory_order_acquire)) return;

        Event e;
        e.name = name;
        e.ph = ph;
        e.ts = timestamp_us();
        e.tid = thread_id_hash();

        // Write event immediately to file instead of buffering
        write_event(e);
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
        std::lock_guard<std::mutex> lk(mutex_);
        if (!outfile_.is_open()) return;

        bool was_first = first_event_.exchange(false);
        if (!was_first) {
            outfile_ << ",";
        }

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
        // Flush every N events to reduce I/O overhead while keeping streaming benefits
        if (++event_count_ % 100 == 0) {
            outfile_.flush();
        }
    }

    std::atomic<bool> started_;
    std::atomic<bool> first_event_;
    Clock::time_point start_time_;
    std::ofstream outfile_;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, std::vector<Event>*> thread_buffers_; // Keep for cleanup only
    uint64_t event_count_{0};
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
