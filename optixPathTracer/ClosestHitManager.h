#ifndef CLOSEST_HIT_MANAGER
#define CLOSEST_HIT_MANAGER

#include <string.h>
#include <optixu/optixpp_namespace.h>

class ClosestHitManager
{
public:
    enum closest_hit_program
    {
        closest_hit_debug,                  // A single closest hit program that contains all (debug) render modes for easy testing. (See debug_render_mode below.)
        closest_hit_virtual_texture,        // A closest hit program corresponding to mode_virtual_texture for testing render speed (frame times).
        closest_hit_atlased_texture_scaled, // A closest hit program corresponding to mode_atlased_texture_scaled for testing render speed (frame times).
        closest_hit_count
    };

    // Render modes for the debug closest hit program
    enum debug_render_mode
    {
        mode_diffuse_colour,
        mode_barycentric_coordinates,
        mode_uv_coordinates,
        mode_classic_texture,
        mode_atlased_uv_coordinates,
        mode_atlased_texture,
        mode_atlased_texture_scaled,
        mode_atlased_texture_scaled_mipmapped,
        mode_differential_footprint,
        mode_atlased_differential_footprint,
        mode_atlased_differential_footprint_a,
        mode_atlased_differential_footprint_b,
        mode_mip_id_tex,
        mode_mip_id_calc,
        mode_tile_id,
        mode_scale_and_bias,
        mode_tile_pool_texture,
        mode_virtual_texture,
        mode_count
    };

    ClosestHitManager(closest_hit_program default_program, debug_render_mode default_mode);
    ~ClosestHitManager();

    closest_hit_program GetProgram();
    void                SetProgram(closest_hit_program program);
    void                NextProgram();
    void                PreviousProgram();
    std::string         ProgramDescription(closest_hit_program program);

    debug_render_mode   GetMode();
    void                SetMode(debug_render_mode mode);
    void                NextMode();
    void                PreviousMode();
    std::string         ModeDescription(debug_render_mode mode);

    bool                IsTileManagementRequired();

private:
    closest_hit_program current_closest_hit_program;
    debug_render_mode   current_render_mode;

    bool mode_requires_vt_tile_management(debug_render_mode mode);
    bool program_requires_vt_tile_management(closest_hit_program program, debug_render_mode mode);
};

#endif // CLOSEST_HIT_MANAGER
