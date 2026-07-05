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

} // namespace backends
} // namespace lemon
