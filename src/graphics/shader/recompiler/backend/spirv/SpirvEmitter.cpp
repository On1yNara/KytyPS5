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
	static const uint32_t loop_guard = [] {
		const char* guard = std::getenv("KYTY_LOOP_GUARD");
		const auto  limit = guard != nullptr ? static_cast<uint32_t>(std::strtoul(guard, nullptr, 0)) : 0u;
		if (limit != 0) {
			char line[96];
			std::snprintf(line, sizeof(line), "Shader recompiler: KYTY_LOOP_GUARD=%u\n", limit);
			Log::WriteToConsoleAndLog(line);
		}
		return limit;
	}();
	if (program.stage != ShaderType::Mesh) state.loop_guard_limit = loop_guard;
	// PPSA02530's persistent GPU-culling compute shader walks its Hi-Z footprint with float loop
	// counters (x += 1.0 / x += 8.0 while x <= max). When a projected bound comes out infinite
	// the counter stops changing and the loop never ends, hanging the GPU (device lost). Give
	// this one shader a loop budget unless KYTY_LOOP_GUARD already set one.
	if (state.loop_guard_limit == 0 && program.shader_hash == 0x8556beb2a0abb67aull) {
		state.loop_guard_limit = 200000u;
		static std::atomic_bool logged = false;
		if (!logged.exchange(true)) {
			Log::WriteToConsoleAndLog(
			    "Shader recompiler: loop guard 200000 for CS 0x8556beb2a0abb67a\n");
		}
	}
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
