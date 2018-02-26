#include "ClosestHitManager.h"

ClosestHitManager::ClosestHitManager(
    closest_hit_program default_program,
    debug_render_mode   default_mode
):
    current_closest_hit_program(default_program),
    current_render_mode(default_mode)
{}

ClosestHitManager::~ClosestHitManager() {}

ClosestHitManager::closest_hit_program ClosestHitManager::GetProgram()
{
    return current_closest_hit_program;
}

void ClosestHitManager::SetProgram(ClosestHitManager::closest_hit_program program)
{
    current_closest_hit_program = program;
}

void ClosestHitManager::NextProgram()
{
    SetProgram(closest_hit_program((GetProgram() + 1) % closest_hit_count));
}

void ClosestHitManager::PreviousProgram()
{
    SetProgram(closest_hit_program((GetProgram() + closest_hit_count - 1) % closest_hit_count));
}

std::string ClosestHitManager::ProgramDescription(closest_hit_program program)
{
    std::string program_descriptions[ClosestHitManager::closest_hit_count] = {
        "Debugging",
        "Virtual Texture",
        "Classic (Scaled) Texture Atlas"
    };
    return program_descriptions[program];
}

ClosestHitManager::debug_render_mode ClosestHitManager::GetMode()
{
    return current_render_mode;
}

void ClosestHitManager::SetMode(debug_render_mode mode)
{
    current_render_mode = mode;
}

void ClosestHitManager::NextMode()
{
    SetMode(debug_render_mode((GetMode() + 1) % mode_count));
}

void ClosestHitManager::PreviousMode()
{
    SetMode(debug_render_mode((GetMode() + mode_count - 1) % mode_count));
}

std::string ClosestHitManager::ModeDescription(debug_render_mode mode)
{
    std::string mode_descriptions[ClosestHitManager::mode_count] = {
        "Single colour diffuse",
        "Barycentric coordinates",
        "Interpolated UV coordinates",
        "Classic textures",
        "Atlased interpolated UV coordinates",
        "Atlased texture (original atlas)",
        "Atlased texture (vt scale atlas)",
        "Atlased texture (vt scale and mipmapped atlas)",
        "Differential T vector",
        "Atlased differential T vector",
        "Atlased differential footprint (component A)",
        "Atlased differential footprint (component B)",
        "Mip ID as colours (mipID from texture)",
        "Mip ID as colours (calculated mipID)",
        "Tile ID",
        "Scale and bias (from page table texture)",
        "Tile pool as texture",
        "Virtual (atlased) texture"
    };
    return mode_descriptions[mode];
}

bool ClosestHitManager::IsTileManagementRequired()
{
    return program_requires_vt_tile_management(current_closest_hit_program, current_render_mode);
}

bool ClosestHitManager::mode_requires_vt_tile_management(debug_render_mode mode)
{
    return mode == mode_scale_and_bias
        || mode == mode_virtual_texture;
}

bool ClosestHitManager::program_requires_vt_tile_management(closest_hit_program program, debug_render_mode mode)
{
    return program == closest_hit_virtual_texture
        || (program == closest_hit_debug && mode_requires_vt_tile_management(mode));
}
