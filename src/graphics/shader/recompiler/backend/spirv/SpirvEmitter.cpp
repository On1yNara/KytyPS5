#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

// Lanes of one RDNA wave run in lockstep, so games let a lane read LDS that another lane of the
// same wave just wrote without any barrier. Host invocations do not run in lockstep (NVIDIA's
// independent thread scheduling in particular), so without a barrier the read can see stale data;
// Vulkan validation reports these as shared-memory data races and the stale values can leave a
// shader looping forever (device lost). Barriers are only legal where control flow is uniform
// across their scope: IR blocks are uniform per guest wave, so the scope is the workgroup when
// it is a single wave, or the host subgroup when it holds exactly one guest wave. Limited to
// 32-wide hosts (NVIDIA, Intel), where the problem shows up; AMD hosts keep the old code.
// Same approach as AnyPS5's recompiler. KYTY_NO_WAVE_LDS_BARRIERS=1 turns it off for comparison.
uint32_t WaveLdsScope(const IR::Program& program, const ShaderWorkgroupInputInfo* workgroup,
                      uint32_t lane_count) {
	static const bool disabled = [] {
		const bool off = std::getenv("KYTY_NO_WAVE_LDS_BARRIERS") != nullptr;
		if (off) Log::WriteToConsoleAndLog("Shader recompiler: KYTY_NO_WAVE_LDS_BARRIERS on\n");
		return off;
	}();
	if (disabled || program.stage != ShaderType::Compute || workgroup == nullptr ||
	    workgroup->host_subgroup_size != 32u) {
		return 0;
	}
	bool writes = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = IR::SharedAccessOf(inst.GetOpcode());
			writes |= access != IR::SharedAccess::None && access != IR::SharedAccess::Read;
		}
	}
	if (!writes) return 0;
	const uint32_t threads = std::max(workgroup->threads_num[0], 1u) *
	                         std::max(workgroup->threads_num[1], 1u) *
	                         std::max(workgroup->threads_num[2], 1u);
	// AnyPS5 found that on NVIDIA a subgroup-scope barrier did not make one lane's LDS writes
	// visible to the others in a single-wave workgroup, while the workgroup scope does.
	if (threads <= program.wave_size) return spv::ScopeWorkgroup;
	// A wave64 program runs two lanes per invocation on a 32-wide host, so one guest wave is one
	// host subgroup; so is a wave32 program.
	if (lane_count == 2u || program.wave_size == workgroup->host_subgroup_size) {
		return spv::ScopeSubgroup;
	}
	return 0;
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const Profile& profile,
                                  bool emit_debug_info) {
	using namespace Emitter;

	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, profile);
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	state.wave_lds_scope = WaveLdsScope(program, workgroup, state.lane_count);
	// Same idea as AnyPS5's APS5_LOOP_GUARD. Mesh shaders are skipped: their SPIR-V 1.4 entry point
	// would have to list the counter variable in its interface.
	// Loop guard (see EmitterState::loop_guard_limit). Persistent GPU-culling compute shaders (as in
	// PPSA02530) walk Hi-Z footprints with float loop counters; with an infinite projected bound
	// the counter stops advancing and the loop never ends, which hangs the GPU (device lost). By
	// default compute shaders get a budget of 5000 evaluated loop exits per invocation, far
	// beyond any finite loop they run. KYTY_LOOP_GUARD=<n> sets the budget for every stage, and
	// KYTY_LOOP_GUARD=0 turns the guard off.
	static const int64_t loop_guard_env = [] {
		const char* guard = std::getenv("KYTY_LOOP_GUARD");
		if (guard == nullptr) return int64_t {-1};
		const auto limit = static_cast<int64_t>(std::strtoul(guard, nullptr, 0));
		char line[96];
		std::snprintf(line, sizeof(line), "Shader recompiler: KYTY_LOOP_GUARD=%lld\n",
		              static_cast<long long>(limit));
		Log::WriteToConsoleAndLog(line);
		return limit;
	}();
	constexpr uint32_t DefaultComputeLoopGuard = 5000u;
	if (program.stage != ShaderType::Mesh) {
		if (loop_guard_env >= 0) {
			state.loop_guard_limit = static_cast<uint32_t>(loop_guard_env);
		} else if (program.stage == ShaderType::Compute) {
			state.loop_guard_limit = DefaultComputeLoopGuard;
			static std::atomic_bool logged = false;
			if (!logged.exchange(true)) {
				Log::WriteToConsoleAndLog(
				    "Shader recompiler: compute loop guard 5000 (KYTY_LOOP_GUARD=0 disables)\n");
			}
		}
	}
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
