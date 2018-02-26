/*
 * Path Traced Virtual Textures (PTVT)
 * Copyright 2018 Hans Cronau
 *
 * File based on the Optix SDK optixPathTracer sample,
 * Copyright 2016 Nvidia Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
// optixPathTracer: simple interactive path tracer
//
//-----------------------------------------------------------------------------

#pragma region Includes

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fstream>             // Used for data collection.
#include <ctime>               // Used for timestamping.
#include <sstream>             // Used for timestamping.
#include <iomanip>             // Used for timestamping.

#include <IL/il.h>             // DevIL (image library).

#include "sampleConfig.h"
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include "optixPathTracer.h"
#include <sutil.h>
#include <Arcball.h>
//#include <PPMLoader.h>       // Replaced by custom ILLoader.
#include "ILLoader.h"          // DevIL Image Loader. Custom replacement of PPMLoader.
//#include <ImageLoader.h>     // Customised to use ILLoader instead of PPMLoader. No longer required.
#include <OptiXMesh.h>         // Required for (customised) loading and use of meshes.

#include "ptvtConfig.h"        // Config file for PTVT project.
#include "ClosestHitManager.h" // Manages render modes for debugging and research purposes.
#include "vtAtlasXML.h"        // Parses XML file containing position and size of (virtual) texture atlas subtextures. Used for updating mesh UVs.
#include "vtHelpers.h"         // VT helper functions.
#include "vtManager.h"         // Keeps track of and manages tiles.
#include "vtMipIDTexture.h"    // Stores mipID of each mip level.
#include "vtTileCache.h"       // Uploads tiles to GPU.
#include "vtTileReader.h"      // Reads tiles from disk.
#include "vtTilesXML.h"

using namespace optix;

#pragma endregion

#pragma region Globals
//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

const char* const SAMPLE_NAME = "optixPathTracer";
const char* const WINDOW_NAME = "Virtual Textures for Path Tracing";

// Path tracer settings
Context        context = 0;
uint32_t       width  = PT_DEFAULT_OUTPUT_WIDTH;
uint32_t       height = PT_DEFAULT_OUTPUT_HEIGHT;
bool           use_pbo = true;

int            frame_number = 1;
int            sqrt_num_samples = 2;
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;      // For parallelograms
Program        pgram_bounding_box = 0;      // For parallelograms
Program        pgram_mesh_intersection = 0; // For meshes
Program        pgram_mesh_bounding_box = 0; // For meshes

// Support dynamic changing of closest hit program
Program closest_hit_programs[ClosestHitManager::closest_hit_count];
OptiXMesh mesh;   // Access to Sponza materials
Material diffuse; // Access to Cornell box materials

// Camera state
float3         camera_eye;
float          camera_pitch;
float          camera_yaw;
float          camera_focal_length;
float          camera_fov;
float3         camera_up;
float3         camera_forward;
float3         camera_lookat;
float3         camera_u;
float3         camera_v;
float3         camera_w;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
float          camera_keyboard_move_speed;
float          camera_mouse_move_speed;
float		   camera_rotate_speed;
enum camera_control_mode { camera_pitch_yaw_controlled, camera_forward_controlled };
camera_control_mode camera_control;

// Mipmap state
float		   footprint_scale;
float		   footprint_bias;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Game state
float		   delta_time;
double		   prev_time;

// Globals - Rendering
const unsigned int tile_determination_depth    = VT_TILE_DETERMINATION_DEPTH;   // Max tile determination depth. (Does NOT determine memory allocation for differentials. See differentialDepth instead.)
const unsigned int unique_tileID_buffer_length = VT_UNIQUE_TILID_BUFFER_LENGTH; // Length of the one dimensional unique tileID buffer. Should be more than (expected) max nr visible tiles per frame. In a rasterizer this buffer can be as small as 4kB (Hollemeersch2010).
const unsigned int differentialDepth           = VT_DIFFERENTIAL_DEPTH;         // Max path depth at which new differentials are created. More differentials linearly take up more memory.

// Global const values - Per virtual texture
const unsigned int min_tile_mipID             = VT_MIN_TILE_MIPID;
const unsigned int tile_pool_texels_wide      = VT_TILE_POOL_TEXELS_WIDE;       // Dimensions in texels. Please only use a multitude of tile_size and preferrebly a power of 2.
const unsigned int keep_alive_time            = VT_KEEP_ALIVE_TIME;             // Tiles are kept 'alive' for 4 frames after they have become invisible.
const unsigned int max_tile_uploads_per_frame = VT_MAX_TILE_UPLOADS_PER_FRAME;  // Max nr of tiles to be uploaded between frames. Prevents system getting stuck when suddenly many tiles become visible.

// Global const values - Scenery
const std::string obj_filename       = "sponza.obj";      // Scene object
const std::string atlas_xml_filename = "atlas.xml";       // Texture atlas xml file
const std::string tiles_xml_filename = "tile_info.xml";   // XML file with info about tile files.
const std::string tiles_folder       = "tiles";           // Folder to find tile files in.
const std::string atlases_folder     = "mipmapped_atlas"; // Folder containing vt scaled texture atlases.

// Global variables
unsigned int frameID;
const unsigned int first_frameID = 1u;

// Debugging variables
std::string vt_debug_run_timestamp;

#ifdef VT_DEBUG_PRINT_RENDER_LOOP
bool vt_debug_print_render_loop = true;
#else
bool vt_debug_print_render_loop = false;
#endif

#ifdef VT_DEBUG_PRINT_TILEIDS
bool vt_debug_print_tileids = true;
#else
bool vt_debug_print_tileids = false;
#endif

#ifdef VT_DEBUG_TILE_BORDERS
bool vt_debug_tile_borders = true;
#else
bool vt_debug_tile_borders = false;
#endif

#ifdef VT_DEBUG_DIRECT_HIT_ONLY
const unsigned int debug_default_limit_to_one_bounce = 1;
#else
const unsigned int debug_default_limit_to_one_bounce = 0;
#endif

#ifdef VT_WRITE_FRAMES_TO_FILES
bool vt_write_frames_to_files = true;
#else
bool vt_write_frames_to_files = false;
#endif

#ifdef VT_WRITE_TILE_COUNTS_TO_FILE
bool vt_write_tile_counts_to_file = true;
#else
bool vt_write_tile_counts_to_file = false;
#endif

#ifdef VT_WRITE_FRAME_TIMES_TO_FILE
bool vt_write_frame_times_to_file = true;
#else
bool vt_write_frame_times_to_file = false;
#endif

// Debugging variables
float variable_step = 0;
enum variable_name
{
	footprint_scale_name,
	footprint_bias_name,
	camera_focal_length_name,
	camera_fov_name,
	camera_keyboard_move_speed_name,
	camera_mouse_move_speed_name,
	camera_rotate_speed_name,
	variable_name_count
};
variable_name current_variable;
std::string variable_name_strings[variable_name_count]
{
	"footprint scale",
	"footrpint bias",
	"camera focal length",
	"camera field of view",
	"camera keyboard move speed",
	"camera mouse move speed",
	"camera rotate speed"
};
float* get_variable(variable_name variable_name)
{
	switch (variable_name)
	{
	case footprint_scale_name:
		return &footprint_scale;
        break;
    case footprint_bias_name:
        return &footprint_bias;
        break;
	case camera_focal_length_name:
		return &camera_focal_length;
		break;
	case camera_fov_name:
		return &camera_fov;
		break;
	case camera_keyboard_move_speed_name:
		return &camera_keyboard_move_speed;
		break;
	case camera_mouse_move_speed_name:
		return &camera_mouse_move_speed;
		break;
	case camera_rotate_speed_name:
		return &camera_rotate_speed;
		break;
	default:
		break;
	}
}

// Render settings
ClosestHitManager closestHitManager(ClosestHitManager::closest_hit_debug, ClosestHitManager::mode_classic_texture);

// Virtual texture specific variables
vtAtlasXML* atlas;                          // Reads and holds info on texture atlas (= virtual texture).
vtTilesXML* tiles;                          // Reads and holds info on virtual texture tiles.
vtTileReader* tileReader;                   // Reads tiles from disk.
vtManager* virtualTextureManager = nullptr; // Keeps track of tiles status on CPU.
vtTileCache* tileCache = nullptr;           // Manages and uploads tiles in/to the tile cache texture.
vtPageTable* pageTable;                     // Contains address data for virtual to physical address translation.
vtMipIDTexture* mipIDTexture;               // Contains mipID data.

// Texture atlases
TextureSampler textureAtlas;     // An unscaled classic texture atlas. (For comparison.)
TextureSampler textureAtlasMip0; // A scaled classic texture atlas.
TextureSampler textureAtlasMip1; // A scaled classic texture atlas.
TextureSampler textureAtlasMip2; // A scaled classic texture atlas.
TextureSampler textureAtlasMip3; // A scaled classic texture atlas.
TextureSampler textureAtlasMip4; // A scaled classic texture atlas.
TextureSampler textureAtlasMip5; // A scaled classic texture atlas.

// Tile data collecting cvs file.
std::ofstream tile_counts_stream;
// Research - Collecting frame times to cvs file.
std::vector<float> frame_times;

#pragma endregion

#pragma region Forward declarations
//------------------------------------------------------------------------------
//
// Forward declarations
//
//------------------------------------------------------------------------------

// PT_Helpers
std::string ptxPath(const std::string& cuda_file);
Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void atlasOptiXMeshMaterials(const Mesh& mesh, OptiXMesh& optixMesh); // For atlasing meshes.
void updateMaterialsClosestHitProgram();
void loadGeometry();
void setupCamera();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);

// VT_Helpers
std::string dataPath(const std::string& data_file);
std::string texturePath(const std::string& texture_file);
void setVirtualTextureVariables(GeometryInstance& gi, bool set_default_texture_atlas_ST);
std::string getTimeStamp();   // Timestamp function for testing purposes.
void initVirtualTexture();    // Bundles VT-related initialisation code.
void resetVirtualTexture();   // Resets virtual texture manager (data structures).
void destroyVirtualTexture(); // Destroys up VT-related stuff.
void initPathDifferentials(); // Path differentials are required for mipmap level determination in VT tile determination and VT rendering.
void printRenderModeChanged();
void printLimitModeChanged(unsigned int limit_to_one_bounce);
void printVirtualTextureUsage();

#pragma endregion

#pragma region Helper functions
//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------


#pragma region PT_Helpers

std::string ptxPath(const std::string& cuda_file)
{
    return
        std::string(sutil::samplesPTXDir()) +
        "/" + std::string(SAMPLE_NAME) + "_generated_" +
        cuda_file +
        ".ptx";
}


Buffer getOutputBuffer()
{
    return context["output_buffer"]->getBuffer();
}


void destroyContext()
{
    if (context)
    {
        context->destroy();
        context = 0;
    }
}

void terminateProgram()
{
    destroyVirtualTexture();
    destroyContext();
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( terminateProgram ); // this function is freeglut-only
#else
    atexit( terminateProgram );
#endif
}


void setMaterial(
    GeometryInstance& gi,
    Material material,
    const std::string& texture_name, // Support texture names
    const std::string& color_name,
    const float3& color)
{
    gi->addMaterial(material);
    setVirtualTextureVariables(gi, true);
    gi["Kd_map"]->setTextureSampler(loadILTexture(context, texturePath(texture_name), make_float3(1, 1, 1)));
    gi[color_name]->setFloat(color);
}


GeometryInstance createParallelogram(
    const float3& anchor,
    const float3& offset1,
    const float3& offset2)
{
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount(1u);
    parallelogram->setIntersectionProgram(pgram_intersection);
    parallelogram->setBoundingBoxProgram(pgram_bounding_box);

    float3 normal = normalize(cross(offset1, offset2));
    float d = dot(normal, anchor);
    float4 plane = make_float4(normal, d);

    // Debug - Ortogonal vector projection onto offset1 and offset2:
    // v1 = normalised(offset1) / magnitude(offset1)
    // = offset1 / magnitude(offset1) / magnitude(offset1)
    // = offset1 / magnitude(offset1)^2
    // = offset1 / sqrt(offset1.x^2 + offset1.y^2 + offset1.z^2)^2
    // = offset1 / offset1.x^2 + offset1.y^2 + offset1.z^2
    // = offset1 / dot(offset1,offset1)
    float3 v1 = offset1 / dot(offset1, offset1);
    float3 v2 = offset2 / dot(offset2, offset2);

    parallelogram["plane"]->setFloat(plane);
    parallelogram["anchor"]->setFloat(anchor);
    parallelogram["v1"]->setFloat(v1);
    parallelogram["v2"]->setFloat(v2);
    parallelogram["offset1"]->setFloat(offset1);
    parallelogram["offset2"]->setFloat(offset2);

    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry(parallelogram);
    return gi;
}


void createContext()
{
    context = Context::create();
    context->setRayTypeCount(2);
    context->setEntryPointCount(2); // 0) path tracer, 1) virtual texture tile determination
    context->setStackSize(1800);

    context["scene_epsilon"]->setFloat(1.e-3f);
    context["pathtrace_ray_type"]->setUint(0u);
    context["pathtrace_shadow_ray_type"]->setUint(1u);
    context["rr_begin_depth"]->setUint(rr_begin_depth);

    // Prepare custom rendering modes
    context["limit_to_one_bounce"]->setUint(debug_default_limit_to_one_bounce);
    context["debug_render_mode"]->setUint(closestHitManager.GetMode());

    // Setup output buffer
    Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["output_buffer"]->set(buffer);

    // Setup programs
    const std::string cuda_file = std::string(SAMPLE_NAME) + ".cu";
    const std::string ptx_path = ptxPath(cuda_file);
    context->setRayGenerationProgram(0, context->createProgramFromPTXFile(ptx_path, "pathtrace_camera"));
    context->setExceptionProgram(0, context->createProgramFromPTXFile(ptx_path, "exception"));
    context->setMissProgram(0, context->createProgramFromPTXFile(ptx_path, "miss"));

    context["sqrt_num_samples"]->setUint(sqrt_num_samples);
    context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in progressive rendering.
    context["bg_color"]->setFloat(make_float3(0.0f));
}


void atlasOptiXMeshMaterials(const Mesh& mesh, OptiXMesh& optixMesh)
{
    vtAtlasXML atlas_xml = vtAtlasXML(dataPath(atlas_xml_filename));
    const unsigned int materialCount = optixMesh.geom_instance->getMaterialCount();
    for (unsigned int i = 0; i < optixMesh.geom_instance->getMaterialCount(); i++)
    {
        Material material = optixMesh.geom_instance->getMaterial(i);

        // Default scale
        float sx = 1.0f;
        float sy = 1.0f;
        // Default offset (translate)
        float tx = 0.0f;
        float ty = 0.0f;
        // Update ST based on subtexture name.
        atlas_xml.subtextureNameToST(mesh.mat_params[i].Kd_map, sx, sy, tx, ty);
        // Set ST on GeometryGroup associated with this texture.
        material["texture_atlas_ST"]->setFloat(sx, sy, tx, ty);
    }
}


void updateMaterialsClosestHitProgram()
{
    // Update diffuse material (used by Cornell Box).
    diffuse->setClosestHitProgram(0, closest_hit_programs[closestHitManager.GetProgram()]);

    // Update mesh materials (used by Sponza).
    if (mesh.geom_instance)
    {
        for (unsigned int i = 0; i < mesh.geom_instance->getMaterialCount(); i++)
        {
            mesh.geom_instance->getMaterial(i)->setClosestHitProgram(0, closest_hit_programs[closestHitManager.GetProgram()]);
        }
    }
}


void loadGeometry()
{
    // Set up material
    const std::string cuda_file = std::string(SAMPLE_NAME) + ".cu";
    std::string ptx_path = ptxPath(cuda_file);
    diffuse = context->createMaterial();
    closest_hit_programs[ClosestHitManager::closest_hit_debug] = context->createProgramFromPTXFile(ptx_path, "diffuse");
    closest_hit_programs[ClosestHitManager::closest_hit_virtual_texture] = context->createProgramFromPTXFile(ptx_path, "virtual_diffuse");
    closest_hit_programs[ClosestHitManager::closest_hit_atlased_texture_scaled] = context->createProgramFromPTXFile(ptx_path, "atlased_diffuse");
    Program diffuse_ah = context->createProgramFromPTXFile(ptx_path, "shadow");

    diffuse->setClosestHitProgram(0, closest_hit_programs[closestHitManager.GetProgram()]);
    diffuse->setAnyHitProgram(1, diffuse_ah);

    Material diffuse_light = context->createMaterial();
    Program diffuse_em = context->createProgramFromPTXFile(ptx_path, "diffuseEmitter");
    diffuse_light->setClosestHitProgram(0, diffuse_em);

    // Set up parallelogram programs
    ptx_path = ptxPath("parallelogram.cu");
    pgram_bounding_box = context->createProgramFromPTXFile(ptx_path, "bounds");
    pgram_intersection = context->createProgramFromPTXFile(ptx_path, "intersect");

    // Set up traingle mesh programs (special ones that support path differentials)
#ifdef VT_RESEARCH_SPEED_ATLAS
    ptx_path = ptxPath("triangle_mesh.cu");  // sutil_sdk
#else
    ptx_path = ptxPath("vtTriangleMesh.cu"); // custom with support for differentials
#endif
    pgram_mesh_bounding_box = context->createProgramFromPTXFile(ptx_path, "mesh_bounds");
    pgram_mesh_intersection = context->createProgramFromPTXFile(ptx_path, "mesh_intersect");

    // create geometry instances
    std::vector<GeometryInstance> gis;

#ifdef SCENE_CORNELL_BOX
    // Predefine Cornell Box colours
    const float3 white = make_float3(0.8f, 0.8f, 0.8f);
    const float3 green = make_float3(0.05f, 0.8f, 0.05f);
    const float3 red = make_float3(0.8f, 0.05f, 0.05f);
    const float3 light_em = make_float3(15.0f, 15.0f, 5.0f);

    // Floor
    gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 559.2f),
        make_float3(556.0f, 0.0f, 0.0f)));
    setMaterial(gis.back(), diffuse, "cloth.ppm", "diffuse_color", white);

    // Ceiling
    gis.push_back(createParallelogram(make_float3(0.0f, 548.8f, 0.0f),
        make_float3(556.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 559.2f)));
    setMaterial(gis.back(), diffuse, "background_ddn.tif", "diffuse_color", white);

    // Back wall
    gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 559.2f),
        make_float3(0.0f, 548.8f, 0.0f),
        make_float3(556.0f, 0.0f, 0.0f)));
    setMaterial(gis.back(), diffuse, "blue_white_circle.png", "diffuse_color", white);

    // Right wall
    gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, 548.8f, 0.0f),
        make_float3(0.0f, 0.0f, 559.2f)));
    setMaterial(gis.back(), diffuse, "sponza_curtain_blue_diff.tga", "diffuse_color", green);

    // Left wall
    gis.push_back(createParallelogram(make_float3(556.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 559.2f),
        make_float3(0.0f, 548.8f, 0.0f)));
    setMaterial(gis.back(), diffuse, "mspaint.bmp", "diffuse_color", red);

    // Short block
    gis.push_back(createParallelogram(make_float3(130.0f, 165.0f, 65.0f),
        make_float3(-48.0f, 0.0f, 160.0f),
        make_float3(160.0f, 0.0f, 49.0f)));
    setMaterial(gis.back(), diffuse, "minecraft_dirt_side.jpg", "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(290.0f, 0.0f, 114.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(-50.0f, 0.0f, 158.0f)));
    setMaterial(gis.back(), diffuse, "minecraft_dirt_side.jpg", "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(130.0f, 0.0f, 65.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(160.0f, 0.0f, 49.0f)));
    setMaterial(gis.back(), diffuse, "minecraft_dirt_side.jpg", "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(82.0f, 0.0f, 225.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(48.0f, 0.0f, -160.0f)));
    setMaterial(gis.back(), diffuse, "minecraft_dirt_side.jpg", "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(240.0f, 0.0f, 272.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(-158.0f, 0.0f, -47.0f)));
    setMaterial(gis.back(), diffuse, "minecraft_dirt_side.jpg", "diffuse_color", white);

    // Tall block
    gis.push_back(createParallelogram(make_float3(423.0f, 330.0f, 247.0f),
        make_float3(-158.0f, 0.0f, 49.0f),
        make_float3(49.0f, 0.0f, 159.0f)));
    setMaterial(gis.back(), diffuse, "tv.ppm", "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(423.0f, 0.0f, 247.0f),
        make_float3(0.0f, 330.0f, 0.0f),
        make_float3(49.0f, 0.0f, 159.0f)));
    setMaterial(gis.back(), diffuse, "tv.ppm", "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(472.0f, 0.0f, 406.0f),
        make_float3(0.0f, 330.0f, 0.0f),
        make_float3(-158.0f, 0.0f, 50.0f)));
    setMaterial(gis.back(), diffuse, "tv.ppm", "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(314.0f, 0.0f, 456.0f),
        make_float3(0.0f, 330.0f, 0.0f),
        make_float3(-49.0f, 0.0f, -160.0f)));
    setMaterial(gis.back(), diffuse, "tv.ppm", "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(265.0f, 0.0f, 296.0f),
        make_float3(0.0f, 330.0f, 0.0f),
        make_float3(158.0f, 0.0f, -49.0f)));
    setMaterial(gis.back(), diffuse, "tv.ppm", "diffuse_color", white);

    // Above GeometryInstances are also shadowers. Add them to a shadow GeometryGroup.
    GeometryGroup cornell_box_shadow_group = context->createGeometryGroup(gis.begin(), gis.end());
    cornell_box_shadow_group->setAcceleration(context->createAcceleration("Trbvh"));

#endif

#ifdef SCENE_SPONZA
    // Predefine Sponza colours
    const float3 sunlight_em = make_float3(255.0f, 255.0f, 251.0f) / 255 * 30;
    
    // Create OptiXMesh and set attributes for virtual textures
    mesh.context = context;
    mesh.intersection = pgram_mesh_intersection;
    mesh.bounds = pgram_mesh_bounding_box;
    mesh.closest_hit = closest_hit_programs[closestHitManager.GetProgram()];
    // Fun fact: leave out above line and the program will crash, displaying a message that the OptiX context is invalid due to "lights" being assigned a 64 byte type instead of 32 byte.
    mesh.any_hit = diffuse_ah;

    // Prepare a function that will process the mesh materials to support texture atlasing.
    OptiXMeshPostFunction post_func = &atlasOptiXMeshMaterials;

    // Load the Sponza mesh data into the OptiXMesh (and post process materials for atlasing).
    loadMesh(dataPath(obj_filename), mesh, optix::Matrix4x4::identity(), post_func);

    // Set gloabal virtual texture variables to geometry.
    setVirtualTextureVariables(mesh.geom_instance, false);

    // Emission_color may sometimes be required by closest hit program.
    mesh.geom_instance["emission_color"]->setFloat(sunlight_em);

    // Add sponza geometry to a GeometryGroup, which is added to the scene further down the line.
    GeometryGroup sponza_geometry_group = context->createGeometryGroup();
    sponza_geometry_group->addChild(mesh.geom_instance);
    sponza_geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
#endif


    // Combine shadow groups to create top_shadower.
    Group top_shadow_group = context->createGroup();
#ifdef SCENE_CORNELL_BOX
    top_shadow_group->addChild(cornell_box_shadow_group);
#endif
#ifdef SCENE_SPONZA
    top_shadow_group->addChild(sponza_geometry_group);
#endif
    top_shadow_group->setAcceleration(context->createAcceleration("Trbvh"));
    context["top_shadower"]->set(top_shadow_group);


    // Light Geometry

#ifdef SCENE_CORNELL_BOX
    // Light in Cornell Box
    gis.push_back(createParallelogram(make_float3(343.0f, 548.6f, 227.0f),
        make_float3(-130.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 105.0f)));
    setMaterial(gis.back(), diffuse_light, "pool_8.ppm", "emission_color", light_em);
#endif

#ifdef SCENE_SPONZA
    // Light for Sponza
    gis.push_back(
        createParallelogram(
        make_float3(343.0f, 2548.6f, 227.0f),
        make_float3(-5300.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 5050.0f)
        )
        );
    setMaterial(gis.back(), diffuse_light, "pool_8.ppm", "emission_color", sunlight_em);
#endif


    // Create top geometry group
    GeometryGroup geometry_instances_group = context->createGeometryGroup(gis.begin(), gis.end()); // Includes all parallelograms: lights and Cornell Box if enabled.
    geometry_instances_group->setAcceleration(context->createAcceleration("Trbvh"));

    // Combine groups and make top_object
    Group top_geometry_group = context->createGroup();
    top_geometry_group->addChild(geometry_instances_group);
#ifdef SCENE_SPONZA
    top_geometry_group->addChild(sponza_geometry_group);
#endif
    top_geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
    context["top_object"]->set(top_geometry_group);


    // Prepare ParallelogramLight buffer

#ifdef SCENE_CORNELL_BOX
#ifdef SCENE_SPONZA
    const size_t number_of_lights = 2u;
#else
    const size_t number_of_lights = 1u;
#endif
#elif defined SCENE_SPONZA
    const size_t number_of_lights = 1u;
#else
    const size_t number_of_lights = 0u;
#endif

    Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(ParallelogramLight));
    light_buffer->setSize(number_of_lights);
    context["lights"]->setBuffer(light_buffer);

    // Create array of lights.
    ParallelogramLight lights[number_of_lights];
    size_t light_array_index = 0;

#ifdef SCENE_CORNELL_BOX
    // Cornell Box Light (Same coordinates as parallelogram geometry above.)
    lights[light_array_index].corner = make_float3(343.0f, 548.6f, 227.0f);
    lights[light_array_index].v1 = make_float3(-130.0f, 0.0f, 0.0f);
    lights[light_array_index].v2 = make_float3(0.0f, 0.0f, 105.0f);
    lights[light_array_index].normal = normalize(cross(lights[light_array_index].v1, lights[light_array_index].v2));
    lights[light_array_index].emission = light_em;
    light_array_index++;
#endif

#ifdef SCENE_SPONZA
    // Sponza Light (Same coordinates as parallelogram geometry above.)
    lights[light_array_index].corner = make_float3(343.0f, 2548.6f, 227.0f);
    lights[light_array_index].v1 = make_float3(-5300.0f, 0.0f, 0.0f);
    lights[light_array_index].v2 = make_float3(0.0f, 0.0f, 5050.0f);
    lights[light_array_index].normal = normalize(cross(lights[light_array_index].v1, lights[light_array_index].v2));
    lights[light_array_index].emission = sunlight_em;
    light_array_index++;
#endif

    memcpy(light_buffer->map(), &lights, sizeof(lights));
    light_buffer->unmap();
}


void setupCamera()
{
#ifdef SCENE_SPONZA
    camera_eye = make_float3(1161.0f, 306.0f, -36.0f);
#else
    camera_eye = make_float3(278.0f, 273.0f, -900.0f);
#endif
    camera_pitch = 0;
#ifdef SCENE_SPONZA
    camera_yaw = -M_PI_2f;
#else
    camera_yaw = 0;
#endif
    camera_focal_length = 1.0f;
    camera_fov = 35.0f;
    camera_keyboard_move_speed = 200.0f; // world units per frame (keyboard)
    camera_mouse_move_speed = 20.0f;     // world units per pixel screenspace by mouse
    camera_rotate_speed = M_PIf / 50;    // rads turned per pixel screenspace by mouse

    camera_rotate = Matrix4x4::identity();

    // Set context camera variables. (Actual values don't really matter at this point.)
    context["frame_number"]->setUint(frame_number);
    context["eye"]->setFloat(camera_eye);
    context["U"]->setFloat(camera_u);
    context["V"]->setFloat(camera_v);
    context["W"]->setFloat(camera_w);
}

void updateCamera()
{
    const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

    const float3    world_right = make_float3(1, 0, 0);
    const float3    world_up = make_float3(0, 1, 0);
    const float3    world_forward = make_float3(0, 0, 1);
    
    if (camera_control == camera_pitch_yaw_controlled)
    {
        const Matrix4x4 camera_rotate_yaw = Matrix4x4::rotate(camera_yaw, world_up);
        const Matrix4x4 camera_rotate_pitch = Matrix4x4::rotate(camera_pitch, world_right);

        camera_rotate = camera_rotate_yaw * camera_rotate_pitch;
        camera_up = normalize(make_float3(camera_rotate*make_float4(world_up, 0.0f)));
        camera_forward = normalize(make_float3(camera_rotate*make_float4(world_forward, 0.0f)));
    }
    else // camera_control == camera_forward_controlled (Cannot be set by user.)
    {
        const float3 camera_right = -cross(camera_forward, world_up);
        camera_up = normalize(cross(camera_forward, camera_right));
        camera_forward = normalize(camera_forward);
    }
    camera_lookat = camera_eye + camera_forward * camera_focal_length;

    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, camera_fov, aspect_ratio,
        camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

    // Restrict camera_u and camera_v to constant dimensions, so that changing the focal length is actually visible.
    const float camera_viewplane_width = tanf(camera_fov / 2.f / 180.f * M_PIf) * camera_focal_length * 2.f;
    camera_u = normalize(camera_u) * camera_viewplane_width;
    camera_v = normalize(camera_v) * (camera_viewplane_width / aspect_ratio);

    if (camera_changed) // reset accumulation
        frame_number = 1;
    camera_changed = false;

    context["frame_number"]->setUint(frame_number++);
    context[ "eye" ]->setFloat( camera_eye );
    context[ "U" ]->setFloat( camera_u );
    context[ "V" ]->setFloat( camera_v );
	context[ "W" ]->setFloat( camera_w );
}


void glutInitialize(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(WINDOW_NAME);
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow(width, height);

    // register glut callbacks
    glutDisplayFunc(glutDisplay);
    glutIdleFunc(glutDisplay);
    glutReshapeFunc(glutResize);
    glutKeyboardFunc(glutKeyboardPress);
    glutMouseFunc(glutMousePress);
    glutMotionFunc(glutMouseMotion);

    registerExitHandler();

    glutMainLoop();
}

#pragma endregion


#pragma region VT_Helpers

std::string dataPath(const std::string& data_file)
{
    return
        std::string(sutil::samplesDir()) +
        "/" + std::string(SAMPLE_NAME) + "/data/" +
        data_file;
}

std::string texturePath(const std::string& texture_file)
{
    return dataPath(texture_file);
}

void setVirtualTextureVariables(GeometryInstance& gi, bool set_default_texture_atlas_ST)
{
    // Set tile pool, page table, mipmap texture, texture atlas.
    gi["tile_pool"]->setInt((tileCache->TextureSampler())->getId());
    gi["page_table"]->setInt((pageTable->TextureSampler())->getId());
    gi["mipID_texture"]->setInt((mipIDTexture->TextureSampler())->getId());
    
    gi["texture_atlas"]->setInt((textureAtlas->getId()));
    gi["texture_atlas_0"]->setInt((textureAtlasMip0->getId()));
    gi["texture_atlas_1"]->setInt((textureAtlasMip1->getId()));
    gi["texture_atlas_2"]->setInt((textureAtlasMip2->getId()));
    gi["texture_atlas_3"]->setInt((textureAtlasMip3->getId()));
    gi["texture_atlas_4"]->setInt((textureAtlasMip4->getId()));
    gi["texture_atlas_5"]->setInt((textureAtlasMip5->getId()));

    // Set default ST (scale, translate).
    // Note: For meshes ST is set per Material. Since GeometryInstances are queried for variables before Materials are, meshes should NOT set this default ST.
    if (set_default_texture_atlas_ST)
    {
        // Default scale
        const float sx = 1.0f;
        const float sy = 1.0f;
        // Default offset (translate)
        const float tx = 0.0f;
        const float ty = 0.0f;

        gi["texture_atlas_ST"]->setFloat(sx, sy, tx, ty);
    }
}

std::string lead_zeroes(const int& integer, const int& number_of_characters)
{
    std::stringstream stringified_integer; // Use a stringstream to ensure leading zeroes.
    stringified_integer << std::setw(number_of_characters) << std::setfill('0') << integer;
    return stringified_integer.str();
}

// Timestamp function
std::string getTimeStamp()
{
    time_t t = time(0); // Gets current time.
    tm* now = localtime(&t); // Converts to local time incl. attributes like year, month, etc.
    std::string stamp =
        lead_zeroes(now->tm_year + 1900, 4)
        + '-' + lead_zeroes(now->tm_mon + 1, 2)
        + '-' + lead_zeroes(now->tm_mday + 1, 2)
        + '-' + lead_zeroes(now->tm_hour + 1, 2)
        + '-' + lead_zeroes(now->tm_min + 1, 2)
        + '-' + lead_zeroes(now->tm_sec + 1, 2);
    return stamp;
}

void initVirtualTexture()
{
    // Initialise the DevIL.
    ilInit();
    ilEnable(IL_ORIGIN_SET); // Because "We want all images to be loaded in a consistent manner"

    atlas = new vtAtlasXML(dataPath(atlas_xml_filename));
    tiles = new vtTilesXML(dataPath(tiles_xml_filename));
    const unsigned int virtual_texture_tiles_wide = atlas->texels_wide() / tiles->texels_wide();

    // Prepare tile pool on GPU.
    tileCache = new vtTileCache(context, tile_pool_texels_wide, tiles->texels_wide(), vt_debug_tile_borders);

    // Prepare tile reading from disk.
    tileReader = new vtTileReader(dataPath(tiles_folder), tiles->file_extension(), 4, IL_RGBA);

    // Prepare a classic texture atlas for comparison against virtual texture atlas->
    textureAtlas = loadILTexture(context, dataPath(atlas->filename()), make_float3(1.0f, 1.0f, 1.0f));
    textureAtlasMip0 = loadILTexture(context, dataPath(atlases_folder) + "/atlas_0.png", make_float3(1.0f, 1.0f, 1.0f));
    textureAtlasMip1 = loadILTexture(context, dataPath(atlases_folder) + "/atlas_1.png", make_float3(1.0f, 1.0f, 1.0f));
    textureAtlasMip2 = loadILTexture(context, dataPath(atlases_folder) + "/atlas_2.png", make_float3(1.0f, 1.0f, 1.0f));
    textureAtlasMip3 = loadILTexture(context, dataPath(atlases_folder) + "/atlas_3.png", make_float3(1.0f, 1.0f, 1.0f));
    textureAtlasMip4 = loadILTexture(context, dataPath(atlases_folder) + "/atlas_4.png", make_float3(1.0f, 1.0f, 1.0f));
    textureAtlasMip5 = loadILTexture(context, dataPath(atlases_folder) + "/atlas_5.png", make_float3(1.0f, 1.0f, 1.0f));

    // Prepare page table texture on GPU.
    pageTable = new vtPageTable(context, virtual_texture_tiles_wide);

    // Prepare mipID sampling texture. (Similar to page table texutre.)
    mipIDTexture = new vtMipIDTexture(context, virtual_texture_tiles_wide);

    // Prepare a tile manager on CPU.
    virtualTextureManager = new vtManager(tileReader, tileCache, pageTable, virtual_texture_tiles_wide, tiles->texels_wide(), tiles->border_texels_wide(), min_tile_mipID, keep_alive_time, max_tile_uploads_per_frame);

    // Set VT related context variables
    context["tileID_path_depth"]->setUint(tile_determination_depth);
    context["virt_tex_size"]->setUint(atlas->texels_wide());
    context["tile_size"]->setUint(tiles->texels_wide());
    context["min_tile_mipID"]->setUint(min_tile_mipID);
    context["tex_mip_id_of_lowest_mip_id_tile"]->setUint(mipIDForDimensions(tiles->texels_wide())); // Explanation: tex_mip_id and tile_mip_id differ (due to some mip-levels being smaller than tile size). The difference is the texture_mip_id of the lowest mip tile.
    context["max_tile_mip_id"]->setUint(mipIDForDimensions(virtual_texture_tiles_wide));

    // Setup mipmap scale and bias.
    footprint_scale = 1.0f;
    footprint_bias = 0.0f;
    context["footprint_scale"]->setFloat(footprint_scale);
    context["footprint_bias"]->setFloat(footprint_bias);

    //Hans, dingen voor de context launch >>
    float distance_bias = 0;
    float focal_length = length(camera_w); // Hardcoding this value is a hack, used because I have yet to figure out how to get the correct focal length from the camera. Note: length(camera_data.W) does not seem reliable.
    float distance_to_onscreen_ratio = (1.0f + distance_bias) / focal_length;
    context["distance_to_onscreen_ratio"]->setFloat(distance_to_onscreen_ratio);
    // Explanation of distance_to_onscreen_ratio:
    // 1.0f represents a 1 on 1 texel to fragment ratio.
    // A distance_bias is added for situations in which we need to lower overall texture resolutions.
    // Note that by adding this bias, you effectively increase the calculated distance from the camera to any surface, which results in a lower desired resolution for the surface's texture.
    // (1.0f + distance_bias) is devided by the focal_length of the camera, so that later multiplication by a path's distance yields an onscreen texel to fragment ratio.
    // Technically distance_to_onscreen_ratio could also be called "distance to texel to fragment ratio ratio".

    // Console output
    std::cout
        << "Virtual Texture variables:  "
        << "\n - tiles texels wide:     " << tiles->texels_wide()
        << "\n - vt texels wide:        " << atlas->texels_wide()
        << "\n - vt tiles wide:         " << virtual_texture_tiles_wide
        << "\n - vt texel mip levels:   " << numberOfMipLevelsForDimensions(atlas->texels_wide())
        << "\n - vt tile mip levels:    " << numberOfMipLevelsForDimensions(virtual_texture_tiles_wide)
        << "\n - vt mip difference:     " << mipIDForDimensions(tiles->texels_wide()) // same as: numberOfMipLevelsForDimensions(atlas->texels_wide()) - numberOfMipLevelsForDimensions(virtual_texture_tiles_wide)
        << "\n - vt max tile mip id:    " << mipIDForDimensions(virtual_texture_tiles_wide)
        << "\n - min tile mip id:       " << min_tile_mipID
        << "\n - tile pool texels wide: " << tile_pool_texels_wide
        << "\n - tile keep in pool:     " << keep_alive_time
        << "\n - max frame tile upload: " << max_tile_uploads_per_frame
        << "\n - feedback depth:        " << tile_determination_depth
        << "\n" << std::endl;

    // Initialise path differentials
    initPathDifferentials();

    // Create tile determination buffers
    frameID = 0u; // Because first frame is frame 1 and doesn't start until ++FrameID.
    context["frameID"]->setUint(frameID);

    Buffer tileID_to_frameID_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, virtualTextureManager->GetNrTilesTotal());
    context["tileID_to_frameID_buffer"]->set(tileID_to_frameID_buffer);

    Buffer unique_tileID_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT3, unique_tileID_buffer_length); // Input-output buffer required for reset in-between frames.
    uint3* unique_tileID_buffer_data = (uint3*)unique_tileID_buffer->map();
    unique_tileID_buffer_data[0].x = (uint)0; // Reset index 0, which stores number of unique tileIDs.
    unique_tileID_buffer->unmap();
    context["unique_tileID_buffer"]->set(unique_tileID_buffer);

    // Setup tile buffer processing programs/entry points
    std::string vt_path = ptxPath("vtTileDetermination.cu");
    Program list_marked_tiles_kernel = context->createProgramFromPTXFile(vt_path, "list_marked_tiles_kernel");
    context->setRayGenerationProgram(1, list_marked_tiles_kernel);

    if (vt_write_tile_counts_to_file)
    {
        // Prepare output file
        tile_counts_stream.open(vt_debug_run_timestamp + " - research method " + VT_RESEARCH_METHOD + " - tiles.csv");
        tile_counts_stream << "FrameID,Tiles Alive (CPU),Tiles Visible (GPU),Tiles Visible (CPU),Tiles Became Alive (CPU)";
        for (int i = 0; i < virtualTextureManager->GetNrMipLevels(); i++)
        {
            tile_counts_stream << ",Tiles Alive in Mip" << i << " (CPU),Tiles Visible in Mip" << i << " (CPU),Tiles Became Alive in Mip" << i << " (CPU)";
        }
        tile_counts_stream << "\n";
    }

    // Prepare log file
    //debug_log_file.open(vt_debug_run_timestamp + " - debug log.txt");

#ifdef VT_DEBUG_OPTIX_PRINT_ENABLED
    context->setPrintEnabled(true);
    context->setPrintBufferSize(1024);
#endif
}

void dryRunEntryPoints()
{
    std::cout << "Dry running entry point 0...";
    context->launch(0, width, height);

    std::cout << "\b\b\b and 1...";
    Buffer tileID_to_frameID_buffer = context["tileID_to_frameID_buffer"]->getBuffer();
    RTsize tileID_to_frameID_buffer_width;
    tileID_to_frameID_buffer->getSize(tileID_to_frameID_buffer_width);
    context->launch(1, static_cast<unsigned int>(tileID_to_frameID_buffer_width));
    Buffer unique_tileID_buffer = context["unique_tileID_buffer"]->getBuffer();
    uint3* unique_tileID_buffer_data = (uint3*)unique_tileID_buffer->map();
    unique_tileID_buffer_data[0].x = (uint)0; // Reset index 0, which stores number of unique tileIDs.
    unique_tileID_buffer->unmap();
    std::cout << "\b\b  " << std::endl;
}

void resetVirtualTexture()
{
    delete virtualTextureManager;
    const unsigned int virtual_texture_tiles_wide = atlas->texels_wide() / tiles->texels_wide();
    virtualTextureManager = new vtManager(tileReader, tileCache, pageTable, virtual_texture_tiles_wide, tiles->texels_wide(), tiles->border_texels_wide(), min_tile_mipID, keep_alive_time, max_tile_uploads_per_frame);
}

void destroyVirtualTexture()
{
    std::cout << "\nThanks for rendering with Virtual Textures. Have a pleasant day.\n" << std::endl;
    delete atlas;
    delete tiles;
    delete virtualTextureManager;
    delete tileCache;
    delete tileReader;
    delete pageTable;
    delete mipIDTexture;
    if (tile_counts_stream.is_open())
    {
        tile_counts_stream.close();
    }
    ilShutDown();
}

void initPathDifferentials()
{
    // Path Differentials
    // const unsigned int differentialDepth = 4; // Commented out because defined elsewhere.
    Buffer positionDifferentials = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height, differentialDepth * 2); // Times 2 to hold both x and y differentials.
    Buffer directionDifferentials = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, width, height, differentialDepth * 2); // Times 2 to hold both x and y differentials.
    Buffer footprintDifferentials = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT2, width, height, differentialDepth * 2); // Times 2 to hold both x and y differentials.
    context["differentialDepth"]->setUint(differentialDepth);
    context["positionDifferentials"]->set(positionDifferentials);
    context["directionDifferentials"]->set(directionDifferentials);
    context["footprintDifferentials"]->set(footprintDifferentials);
}

void printClosestHitProgramChanged()
{
    std::cout << "Selected closest hit program: " << closestHitManager.ProgramDescription(closestHitManager.GetProgram()) << "." << std::endl;
}

void printRenderModeChanged()
{
    std::cout << "Selected debug render mode: " << closestHitManager.ModeDescription(closestHitManager.GetMode()) << "." << std::endl;
}

void printLimitModeChanged(unsigned int limit_to_one_bounce)
{
    std::string description;
    switch (limit_to_one_bounce)
    {
    case 0:
        description = "Default depth";
        break;
    case 1:
    default:
        description = "One bounce only";
        break;
    }
    std::cout << "Bounce limit " << limit_to_one_bounce << ": " << description << "." << std::endl;
}

void printVirtualTextureUsage()
{
    std::cout
        << "Virtual Texture keys:\n"
        << "  j Previous render mode\n"
        << "  k Toggle bounce limit\n"
        << "  l Next render mode\n"
        << "  i Reinitialise virtual texture\n"
        << "  b Toggle debug red tile edges\n"
        << "  p Toggle debug print render loop info\n"
        << "Recording keys:\n"
        << "  r Record frames to files\n"
        << "  f Save single frame to file\n"
        << "Camera keys:\n"
        << "  c Reset camera orientation\n"
        << "  x Write camera details to console\n"
        << "  w Move forward\n"
        << "  a Move left\n"
        << "  s Move backward\n"
        << "  d Move right\n"
        << "  q Move down\n"
        << "  e Move up\n"
        << "Dynamic variables:\n"
        << "  , Select previous variable\n"
        << "  . Select next variable\n"
        << "  [ Decrese variable step (exponentially)\n"
        << "  [ Increase variable step (exponentially)\n"
        << "  - Subtract variable step from selected variable\n"
        << "  = Add variable step to selected variable\n"
        << std::endl;
}

bool has_suffix(const std::string &string, const std::string &suffix)
{
    return string.size() >= suffix.size() && string.compare(string.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void saveBuffer(const std::string &file_name)
{
	//If a ppm file format was requested, let sutil library take care of this.
	if (has_suffix(file_name, ".ppm"))
	{
		sutil::displayBufferPPM(file_name.c_str(), getOutputBuffer());
	}
	else //  Use sutil and DevIL.
	{
		const std::string ppm_name = file_name + ".ppm";
		sutil::displayBufferPPM(ppm_name.c_str(), getOutputBuffer()); // Exploit sutil by first creating a .ppm file.
		ilImage bufferImage = ilImage(ppm_name.c_str()); // Open that one with DevIL.
		bufferImage.Save(file_name.c_str()); // And save it again in a different format. (You have 2 files now.)
		bufferImage.Delete();
	}
}

#pragma endregion

#pragma endregion

#pragma region Research functions

float frame_lerp(int first_frame, int last_frame, int current_frame)
{
    return ((float)current_frame - (float)first_frame) / ((float)last_frame - (float)first_frame);
}

void PTVT_research_step_scenario_none()
{
    if (frameID == 1)
    {
        std::cout << "No research method was set.\nFeel free to fly around. :)\n" << std::endl;
    }
}

void PTVT_research_step_method_speed()
{
    const int frame_start_stand_still = 1;
    const int frame_start_moving = frame_start_stand_still + 120;
    const int frame_start_moving_back = frame_start_moving + 240;
    const int frame_start_last_stand_still = frame_start_moving_back + 120;
    const int frame_done = frame_start_last_stand_still + 120;

    const float3 camera_start = make_float3(1306.69f, 650.978f, -52.7876f); // viewing hall of Sponza (symmetrically)
    const float3 camera_turn  = make_float3(-1409.57f, 650.978f, -52.7876f); // other side of hall
    const float3 camera_stop  = camera_start; // close to a wall

    const bool debug_test_loop = false;

    // Frames
    if (frameID < frame_start_moving)
    {
        if (frameID == frame_start_stand_still)
        {
            // Test takes control over camera and will set camera_forward
            camera_control = camera_forward_controlled;

            // Camera in starting position
            camera_eye     = camera_start;
            camera_forward = camera_turn - camera_start;
            camera_changed = true;
        }

        // No camera movement
    }
    else if (frameID < frame_start_moving_back)
    {
        // Lerp camera to other side room
        const float lerp_amount = frame_lerp(frame_start_moving, frame_start_moving_back - 1, frameID);
        camera_eye = lerp(camera_start, camera_turn, lerp_amount);
        camera_changed = true;
    }
    else if (frameID < frame_start_last_stand_still)
    {
        if (frameID == frame_start_moving_back)
        {
            // Turn camera around
            camera_forward = camera_stop - camera_turn;
            camera_changed = true;
        }

        // Lerp camera back to first side room
        const float lerp_amount = frame_lerp(frame_start_moving_back, frame_start_last_stand_still - 1, frameID);
        camera_eye = optix::lerp(camera_turn, camera_stop, lerp_amount);
        camera_changed = true;
    }
    else if (frameID < frame_done)
    {
        // No camera movement
    }
    else
    {
        if (debug_test_loop)
        {
            frameID = 0;
            // Camera in starting position
            camera_eye     = camera_start;
            camera_forward = normalize(camera_turn - camera_start);
            camera_changed = true;
        }
        else
        {
            // Test done
            
            // If this is a build created to time frames, write collected frame times to file.
#ifdef VT_WRITE_FRAME_TIMES_TO_FILE
            std::ofstream frame_times_stream;
            frame_times_stream.open(vt_debug_run_timestamp + " - research method " + VT_RESEARCH_METHOD + " - frame_times.csv");
            if (frame_times_stream.is_open())
            {
                frame_times_stream << "FrameID,Frame Time\n";
                unsigned int corresponding_frameID = first_frameID;
                for (const float &frame_time : frame_times)
                {
                    frame_times_stream << corresponding_frameID << "," << frame_time << "\n";
                    corresponding_frameID++;
                }
                frame_times_stream.close();
            }
            else
            {
                std::cout << "Whoops! It seems you want to write frame time statistics to a file that hasn't been succesfully opened!" << std::endl;
            }
#endif

            terminateProgram();
            exit(0);
        }
    }
}

void PTVT_research_step_method_accuracy()
{
    const int frame_start_classical_render = 1;
    const int frame_start_vt_render = frame_start_classical_render + 2048; // also gives vt time to load all tiles that were visible during last 2048 frames
    const int frame_done = frame_start_vt_render + 2048;

    // view of hall Sponza with interesting perspective
    const float3 camera_start_position = make_float3(1004.02f, 676.293f, 140.859f);
    const float  camera_start_pitch    = 0.367071f;
    const float  camera_start_yaw      = -1.94758f;

    const bool debug_test_loop = false;

    // Frames
    if (frameID < frame_start_vt_render)
    {
        // Do classic texture render
        if (frameID == frame_start_classical_render)
        {
            // Camera in starting position
            camera_eye   = camera_start_position;
            camera_pitch = camera_start_pitch;
            camera_yaw   = camera_start_yaw;
            closestHitManager.SetMode(ClosestHitManager::mode_atlased_texture_scaled);
            camera_changed = true;
            std::cout << "Creating classic render..." << std::endl;
        }

        // No camera movement
        std::cout << "\r - Frame " << lead_zeroes(frameID - frame_start_classical_render, 4) << "/" << lead_zeroes(frame_start_vt_render - frame_start_classical_render, 4);
        std::cout << " " << (int)(100*frame_lerp(frame_start_classical_render, frame_start_vt_render, frameID)) << "%";
        
    }
    else if (frameID < frame_done)
    {
        if (frameID == frame_start_vt_render)
        {
            // Store classic render
            saveBuffer(vt_debug_run_timestamp + "_classic_render_" + lead_zeroes(frame_start_vt_render, 4) + "_frames" + VT_WRITE_FRAMES_TYPE);
            std::cout << " - Done and saved." << std::endl;

            // Setup for VT render
            closestHitManager.SetMode(ClosestHitManager::mode_virtual_texture);
            camera_changed = true;
            std::cout << "Creating vt render..." << std::endl;
        }

        // No camera movement
        std::cout << "\r - Frame " << lead_zeroes(frameID - frame_start_vt_render, 4) << "/" << lead_zeroes(frame_done - frame_start_vt_render, 4);
        std::cout << " " << (int)(100*frame_lerp(frame_start_vt_render, frame_done, frameID)) << "%";
    }
    else
    {
        // Store VT render
        saveBuffer(vt_debug_run_timestamp + "_vt_render_" + lead_zeroes(frame_start_vt_render, 4) + "_frames" + VT_WRITE_FRAMES_TYPE);

        if (debug_test_loop)
        {
            frameID = 0;
            // Camera reset
            closestHitManager.SetMode(ClosestHitManager::mode_atlased_texture_scaled);
            camera_changed = true;
        }
        else
        {
            // Test done --> Exit
            terminateProgram();
            exit(0);
        }
    }
}

void (*PTVT_research_step)(void) = PTVT_research_step_scenario_none;

void PTVT_research_init()
{
    vt_debug_run_timestamp = getTimeStamp();

    // Set method-dependent variables
#if defined(VT_RESEARCH_METHOD_SPEED_VISIBLE_TILES)
    PTVT_research_step = PTVT_research_step_method_speed;
    closestHitManager.SetProgram(ClosestHitManager::closest_hit_virtual_texture);
#elif defined(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_ATLASED)
    PTVT_research_step = PTVT_research_step_method_speed;
    closestHitManager.SetProgram(ClosestHitManager::closest_hit_atlased_texture_scaled);
#elif defined(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_VIRTUAL)
    PTVT_research_step = PTVT_research_step_method_speed;
    closestHitManager.SetProgram(ClosestHitManager::closest_hit_virtual_texture);
#elif defined(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_VIRTUAL_NO_TILE_UPLOAD)
    PTVT_research_step = PTVT_research_step_method_speed;
    closestHitManager.SetProgram(ClosestHitManager::closest_hit_virtual_texture);
#elif defined(VT_RESEARCH_METHOD_ACCURACY)
    PTVT_research_step = PTVT_research_step_method_accuracy;
    closestHitManager.SetProgram(ClosestHitManager::closest_hit_debug);
#else
    return;
#endif


    // Console output
    std::cout << "Note: This build was set up to conduct research measurements.\n"
        << "      Free camera movement is turned off.\n" << std::endl;
}

#pragma endregion

#pragma region GLUT
//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{

#ifndef VT_RESEARCH_BUILD
    if (vt_debug_print_render_loop)
    {
        std::cout << "Start of new frame." << std::endl;
    }
#endif

    // Increase the frame counter frameID. Unlike the frame counter frame_number (see above),
    // this one does not reset on camera movement and its purpose it to give every frame a unique number.
    // Its is used in tile determination to mask buffer data.
    // Note: the first frame is frame 1. This prevents unset buffers from accidentally matching the frameID.
    context["frameID"]->setUint(++frameID);

    // If this is the first frame, delta-time is 0.
    delta_time = frameID == first_frameID ? 0.0 : static_cast<float>(sutil::currentTime() - prev_time);
    prev_time = sutil::currentTime();

#if defined(VT_WRITE_FRAME_TIMES_TO_FILE)
    if (frameID >= 2) // Frame time of frame 1 is only known at frame 2. (Frame 0 does not exist.)
    {
        frame_times.push_back(delta_time);
    }
#elif !defined(VT_RESEARCH_BUILD)
    if (vt_write_frame_times_to_file && frameID >= 2) // Frame time of frame 1 is only known at frame 2. (Frame 0 does not exist.)
    {
        frame_times.push_back(delta_time);
    }
#endif

    PTVT_research_step();

#ifndef VT_RESEARCH_BUILD
    // Adjust mipmap level scale and bias if required.
    // TODO(HansCronau): Virtual texture manager could respond to tile pool saturation by adjusting mipmap level bias (could use setMipLevelBiason texture instead of changing closest hit program).
    context["footprint_scale"]->setFloat(footprint_scale);
    context["footprint_bias"]->setFloat(footprint_bias);
#endif

	updateCamera();

	Buffer buffer = context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize(buffer_width, buffer_height);

#if !defined(VT_RESEARCH_BUILD) || defined(VT_RESEARCH_METHOD_ACCURACY)
    context["debug_render_mode"]->setUint(closestHitManager.GetMode());
#endif

#ifndef VT_RESEARCH_BUILD
    if (vt_debug_print_render_loop)
    {
        std::cout << "Launching render pass..." << std::endl;
    }
#endif

	context->launch(0, width, height);

#if   defined(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_ATLASED  )
    if (false)
#elif defined(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_VIRTUAL) || defined(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_VIRTUAL_NO_TILE_UPLOAD)
    if (true)
#else
    if (closestHitManager.IsTileManagementRequired())
#endif
    {

#ifndef VT_RESEARCH_BUILD
	    // Post launch data processing
	    if (vt_debug_print_render_loop)
	    {
		    std::cout << "Launching post processing entry points (for processing tileID data)..." << std::endl;
	    }
#endif

        // Collect visible tileIDs (tiles marked with current frameID) from tileID_to_frameID_buffer, and store in unique_tileID_buffer.
        Buffer tileID_to_frameID_buffer = context["tileID_to_frameID_buffer"]->getBuffer();
        RTsize tileID_to_frameID_buffer_width;
        tileID_to_frameID_buffer->getSize(tileID_to_frameID_buffer_width);
        context->launch(1, static_cast<unsigned int>(tileID_to_frameID_buffer_width));

#ifndef VT_RESEARCH_BUILD
        if (vt_debug_print_render_loop)
        {
            std::cout << "Now post processing on CPU..." << std::endl;
        }
#endif

        // Process unique_tileID_buffer
        virtualTextureManager->PrepareForNewFrame();
        Buffer unique_tileID_buffer = context["unique_tileID_buffer"]->getBuffer();
        uint3* unique_tileID_buffer_data = (uint3*)unique_tileID_buffer->map();
        uint number_of_unique_tileIDs = unique_tileID_buffer_data[0].x;
        unique_tileID_buffer_data[0].x = (uint)0; // Reset index 0, which stores number of unique tileIDs.

        for (unsigned int i = 1; i < unique_tileID_buffer_length && i < number_of_unique_tileIDs + 1; i++) // Start at 1 because first index contains this frame's nr of unique tile IDs.
        {
            virtualTextureManager->RegisterTileVisible(unique_tileID_buffer_data[i].x, unique_tileID_buffer_data[i].y, unique_tileID_buffer_data[i].z);
#ifndef VT_RESEARCH_BUILD
            if (vt_debug_print_tileids)
            {
                std::cout << "- tile(" << unique_tileID_buffer_data[i].x << ", " << unique_tileID_buffer_data[i].y << ", " << unique_tileID_buffer_data[i].z << ")" << std::endl;
            }
#endif
        }

#if !defined(VT_RESEARCH_BUILD) || defined(VT_RESEARCH_METHOD_SPEED_VISIBLE_TILES)
        if (vt_write_tile_counts_to_file)
        {
            if (tile_counts_stream.is_open())
            {
                tile_counts_stream << frameID << "," << virtualTextureManager->GetNrTilesAlive() << "," << number_of_unique_tileIDs << "," << virtualTextureManager->GetNrTilesVisible() << "," << virtualTextureManager->GetNrTilesBecameAlive();
                for (int i = 0; i < virtualTextureManager->GetNrMipLevels(); i++)
                {
                    tile_counts_stream << "," << virtualTextureManager->GetNrTilesAlive(i) << "," << virtualTextureManager->GetNrTilesVisible(i) << "," << virtualTextureManager->GetNrTilesBecameAlive(i);
                }
                tile_counts_stream << "\n";
            }
            else
            {
                std::cout << "Whoops! It seems you want to write tile statistics to a file that hasn't been succesfully opened!" << std::endl;
            }
        }
#endif

#ifndef VT_RESEARCH_BUILD
        if (vt_write_frames_to_files)
        {
            saveBuffer(vt_debug_run_timestamp + "_frame_" + lead_zeroes(frameID, 6) + VT_WRITE_FRAMES_TYPE);
        }
#endif

        unique_tileID_buffer->unmap();
        virtualTextureManager->UpdateVirtualTexture();

#ifndef VT_RESEARCH_BUILD
        if (vt_debug_print_render_loop)
        {
            std::cout << "Now displaying output buffer..." << std::endl;
        }
#endif
    }

    sutil::displayBufferGL( getOutputBuffer() );
    
    {
      static unsigned frame_count = 0;
      sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
    
#ifndef VT_RESEARCH_BUILD
    if (vt_debug_print_render_loop)
    {
        std::cout << "Frame done." << std::endl << std::endl;
    }
#endif
}


void glutKeyboardPress( unsigned char key, int x, int y )
{
#ifndef VT_RESEARCH_BUILD
    switch( key )
    {
        case( 27 ): // ESC
        {
            terminateProgram();
            exit(0);
        }
	// Saving frame(s)
        case 'f':
        {
            const std::string outputImage = vt_debug_run_timestamp + "_frame_" + lead_zeroes(frameID, 6) + VT_WRITE_FRAMES_TYPE;
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            saveBuffer(outputImage);
            break;
		}
		case 'r':
			vt_write_frames_to_files = !vt_write_frames_to_files;
			std::cout << (vt_write_frames_to_files ? "Started" : "Stopped") << " recording frames to " << VT_WRITE_FRAMES_TYPE << " files." << std::endl;
			break;
	// Closest hit program settings
		case 'k':
			context["limit_to_one_bounce"]->setUint((context["limit_to_one_bounce"]->getUint() + 1u) % 2u);
			printLimitModeChanged(context["limit_to_one_bounce"]->getUint());
			camera_changed = true;
			break;
		case 'j':
            closestHitManager.PreviousMode();
			printRenderModeChanged();
			camera_changed = true;
			break;
		case 'l':
            closestHitManager.NextMode();
            printRenderModeChanged();
			camera_changed = true;
			break;
		case 'b':
			vt_debug_tile_borders = !vt_debug_tile_borders;
			tileCache->SetDebugTileBorders(vt_debug_tile_borders);
			resetVirtualTexture();
			camera_changed = true;
			std::cout << (vt_debug_tile_borders ? "Rendering" : "Not rendering") << " red debug borders around tiles." << std::endl;
			break;
		case 'i':
			resetVirtualTexture();
			camera_changed = true;
			std::cout << "Reset virtual texture (CPU data structures)." << std::endl;
			break;
		case 'p':
			vt_debug_print_render_loop = !vt_debug_print_render_loop;
			std::cout << (vt_debug_print_render_loop ? "Start" : "Stopped") << " printing live render loop info." << std::endl;
			break;
	// Camera
		case 'c':
			setupCamera();
			camera_changed = true;
			std::cout << "Camera orientation was reset.\n";
			break;
		case 'x':
			std::cout
				<< "Camera details:\n"
				<< " - focal length:  " << camera_focal_length << "\n"
				<< " - pitch (rads):  " << camera_pitch << "\n"
				<< " - yaw (rads):    " << camera_yaw << "\n"
				<< " - field of view: " << camera_fov << "\n"
				<< " - eye: " << camera_eye.x << ", " << camera_eye.y << ", " << camera_eye.z << "\n"
				<< " - u:   " << camera_u.x << ", " << camera_u.y << ", " << camera_u.z << "\n"
				<< " - v:   " << camera_v.x << ", " << camera_v.y << ", " << camera_v.z << "\n"
				<< " - w:   " << camera_w.x << ", " << camera_w.y << ", " << camera_w.z
				<< std::endl;
			break;
	// Camera movement
        case 'w':
        {
            const float distance = delta_time * camera_keyboard_move_speed;
            camera_eye += normalize(camera_w) * distance;
            camera_changed = true;
            break;
        }
        case 'a':
        {
            const float distance = delta_time * camera_keyboard_move_speed;
            camera_eye -= normalize(camera_u) * distance;
            camera_changed = true;
            break;
        }
        case 's':
        {
            const float distance = delta_time * camera_keyboard_move_speed;
            camera_eye -= normalize(camera_w) * distance;
            camera_changed = true;
            break;
        }
        case 'd':
        {
            const float distance = delta_time * camera_keyboard_move_speed;
            camera_eye += normalize(camera_u) * distance;
            camera_changed = true;
            break;
        }
        case 'q':
        {
            const float distance = delta_time * camera_keyboard_move_speed;
            camera_eye -= make_float3(0,1,0) * distance;
            camera_changed = true;
            break;
        }
        case 'e':
        {
            const float distance = delta_time * camera_keyboard_move_speed;
            camera_eye += make_float3(0,1,0) * distance;
            camera_changed = true;
            break;
		}
	// Dynamic variables
		case '-':
			*get_variable(current_variable) -= pow(2, variable_step);
			camera_changed = true;
			std::cout << "Decreased " << variable_name_strings[static_cast<int>(current_variable)] << ": " << *get_variable(current_variable) << ".\n";
			break;
		case '=':
			*get_variable(current_variable) += pow(2, variable_step);
			camera_changed = true;
			std::cout << "Increased " << variable_name_strings[static_cast<int>(current_variable)] << ": " << *get_variable(current_variable) << ".\n";
			break;
		case '[':
			variable_step -= 1;
			std::cout << "Variable step" << " decreased: " << pow(2, variable_step) << ".\n";
			break;
		case ']':
			variable_step += 1;
			std::cout << "Variable step" << " increased: " << pow(2, variable_step) << ".\n";
			break;
		case ',':
			current_variable = static_cast<variable_name>((static_cast<int>(current_variable)+static_cast<int>(variable_name_count)-1) % static_cast<int>(variable_name_count));
			std::cout << "Selected previous variable: " << variable_name_strings[static_cast<int>(current_variable)] << ".\n";
			break;
		case '.':
			current_variable = static_cast<variable_name>((static_cast<int>(current_variable)+static_cast<int>(variable_name_count)+1) % static_cast<int>(variable_name_count));
			std::cout << "Selected next variable: " << variable_name_strings[static_cast<int>(current_variable)] << ".\n";
			break;
    // Closest hit programs
        case '1':
            closestHitManager.SetProgram(ClosestHitManager::closest_hit_debug);
            updateMaterialsClosestHitProgram();
            camera_changed = true;
            printClosestHitProgramChanged();
            break;
        case '2':
            closestHitManager.SetProgram(ClosestHitManager::closest_hit_virtual_texture);
            updateMaterialsClosestHitProgram();
            camera_changed = true;
            printClosestHitProgramChanged();
            break;
        case '3':
            closestHitManager.SetProgram(ClosestHitManager::closest_hit_atlased_texture_scaled);
            updateMaterialsClosestHitProgram();
            camera_changed = true;
            printClosestHitProgramChanged();
            break;
    }
#endif
}


void glutMousePress( int button, int state, int x, int y )
{
#ifndef VT_RESEARCH_BUILD
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
#endif
}


void glutMouseMotion( int x, int y)
{
#ifndef VT_RESEARCH_BUILD
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ); // Speed should be per pixel, not per screen width.
        const float dy = static_cast<float>( y - mouse_prev_pos.y ); // Speed should be per pixel, not per screne height.
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float distance = dmax * delta_time * camera_mouse_move_speed;
        const float3 movement = normalize(camera_w) * distance;
		camera_eye    += movement;
        camera_changed = true;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
		camera_yaw    += (x - mouse_prev_pos.x) * delta_time * -camera_rotate_speed; // Speed should be per pixel, not per screen width.
		camera_pitch  += (y - mouse_prev_pos.y) * delta_time *  camera_rotate_speed; // Speed should be per pixel, not per screne height.
        camera_pitch   = clamp(camera_pitch, -M_PI_2f, M_PI_2f); // Clamp to prevent camera form turning upside down.
        camera_changed = true;
    }

    mouse_prev_pos  = make_int2( x, y );
#endif
}


void glutResize( int w, int h )
{
#ifndef VT_RESEARCH_BUILD
    if ( w == (int)width && h == (int)height ) return;

    camera_changed = true;

    width  = w;
    height = h;
    
    sutil::resizeBuffer( getOutputBuffer(), width, height );

    // Resize differentials buffers recording differentials per fragment.
    context["positionDifferentials" ]->getBuffer()->setSize(width, height, differentialDepth * 2);
    context["directionDifferentials"]->getBuffer()->setSize(width, height, differentialDepth * 2);
    context["footprintDifferentials"]->getBuffer()->setSize(width, height, differentialDepth * 2);

    glViewport(0, 0, width, height);                                               

    glutPostRedisplay();
#endif
}

#pragma endregion

#pragma region Main
//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help               Print this usage message and exit.\n"
        "  -f | --file               Save single frame to file and exit.\n"
        "  -n | --nopbo              Disable GL interop for display buffer.\n"
        "  -m | --mesh <mesh_file>   Specify path to mesh to be loaded.\n"
        "App Keystrokes:\n"
        "  q  Quit\n" 
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        << std::endl;

    exit(1);
}


int main( int argc, char** argv )
{

#pragma region DEV_DEBUG_TEST_STUFF_AND_SUCH

#ifdef VT_TEST_ADDRESS_TRANSLATION
	optix::float4 sb = createScaleAndBias(2, 0, 3, 8, 2, 3);
	std::cout << std::fixed << std::setprecision(8) << "Scale: " << sb.x << std::endl;
	std::cout << std::fixed << std::setprecision(8) << "BiasX: " << sb.y << std::endl;
	std::cout << std::fixed << std::setprecision(8) << "BiasY: " << sb.z << std::endl;
	std::cin.get();
	optix::float2 uv = devirtualiseAddress(make_float3(sb), make_float2(1.0f / 8.0f, 7.0f / 8.0f));
	std::cout << std::fixed << std::setprecision(8) << "PhysU: " << uv.x << std::endl;
	std::cout << std::fixed << std::setprecision(8) << "PhysV: " << uv.y << std::endl;
	std::cin.get();
	exit(0);
#endif

#ifdef VT_TEST_TILE_READER
	// Note: this test WILL crash if run in the VS IDE, if "build\path_tracer\Lenna.png" is not a valid path.

	// Initialise the DevIL.
	ilInit();
	ilEnable(IL_ORIGIN_SET); // Because "We want all images to be loaded in a consistent manner"

	vtTileReader tileReader = vtTileReader("Lenna.png", tile_size);

	ilImage* tileImage;

	tileImage = tileReader.GetTileImage(0, 0, 0);
	tileImage->Save("first tile.png");
	tileImage->Delete();

	//tileImage = tileReader.GetTileImage(2, 3, 3); // for 128 sized tiles
	//tileImage = tileReader.GetTileImage(1, 1, 1); // for 256 sized tiles
	// Universal:
	uint last_tile_index = numberOfIndicesForDimensions(512 / tile_size) - 1;
	uint mipID, x, y;
	tileIDForOffset(last_tile_index, mipID, x, y);
	tileImage = tileReader.GetTileImage(mipID, x, y); // for tiles max 512 in size

	tileImage->Save("last tile.png");
	tileImage->Delete();

	ilShutDown();
	exit(0);
#endif

#pragma endregion

    std::string out_file;
    std::string mesh_file = std::string( sutil::samplesDir() ) + "/data/cow.obj";
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if( arg == "-m" || arg == "--mesh" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            mesh_file = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        createContext();
        setupCamera();

        // Initialise research scripts.
        PTVT_research_init();

		// Initialise VT
		printVirtualTextureUsage();

		std::cout << "Initialising virtual texture..." << std::endl;
		initVirtualTexture();
		std::cout << "Initialising virtual texture done.\n" << std::endl;

		std::cout << "Loading geometry..." << std::endl;
		loadGeometry();
		std::cout << "Loading geometry done.\n" << std::endl;

		std::cout << "Validating OptiX context..." << std::endl;
		context->validate();
		std::cout << "Validating OptiX contex done.\n" << std::endl;

        std::cout << "Dry run all entry points to prevent slow first frame..." << std::endl;
        dryRunEntryPoints();
        std::cout << "Dry run entry points done.\n" << std::endl;

        if ( out_file.empty() )
        {
			std::cout << "GLUT will now run interactive path tracer...\n" << std::endl;
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            terminateProgram();
        }

        return 0;
    }
    SUTIL_CATCH( context->get() )
}

#pragma endregion
