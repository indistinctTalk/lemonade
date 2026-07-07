#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace lemon {
namespace backends {

struct VLLMArgResolution {
    std::vector<std::string> args;
    bool has_memory_budget_arg = false;
    // True when the user/family already supplied --dtype, so backend code
    // should not force its own (e.g. the AWQ float16 default).
    bool has_dtype_arg = false;
    // True when the user/family explicitly asked for --enforce-eager. The resolver
    // treats it as a managed intent: it is detected here and stripped from `args`
    // (VLLMServer::load() re-emits it from the launch policy, so a raw passthrough
    // would duplicate it). This lets a user force eager execution even on a
    // discrete-HBM GPU that would otherwise default to CUDA graphs — an escape
    // hatch for a model whose graph capture misbehaves.
    bool has_enforce_eager = false;
    bool has_quantization_arg = false;
    std::string quantization_arg;
};

VLLMArgResolution resolve_vllm_args(const std::string& model_name,
                                    const std::string& checkpoint,
                                    const nlohmann::json& config,
                                    const std::string& user_vllm_args);

// Discrete-HBM datacenter GPUs (AMD Instinct, gfx9xx) get vLLM's native memory
// budgeting and graph capture; shared-memory APUs and consumer GPUs keep the
// conservative launch defaults. This predicate is the single seam a future
// hardware/backend-aware memory planner replaces.
bool is_discrete_hbm_arch(const std::string& arch);

// The launch-flag decisions VLLMServer::load() derives from the GPU device class.
// Extracted so the discrete-HBM vs conservative-default wiring is unit-testable
// without building the full args vector or launching the vllm-server subprocess.
struct DeviceClassLaunchPolicy {
    bool enforce_eager;     // push --enforce-eager (disables CUDA-graph capture)
    bool force_awq_kernel;  // force the 'awq' kernel (+ float16) for AWQ models
    bool cap_kv_cache;      // push the fixed --kv-cache-memory-bytes cap
};

DeviceClassLaunchPolicy device_class_launch_policy(const std::string& arch,
                                                   bool has_memory_budget_arg,
                                                   bool has_enforce_eager = false);

} // namespace backends
} // namespace lemon
