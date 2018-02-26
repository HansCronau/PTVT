#include <optix.h>
#include "vtHelpers.h"

using namespace optix;

rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtBuffer<uint, 1>  tileID_to_frameID_buffer;
rtBuffer<uint3, 1> unique_tileID_buffer;
rtDeclareVariable(uint, frameID, , );

// This kernel was based on work by Hollemeersch et al. 2010
RT_PROGRAM void list_marked_tiles_kernel()
{
	// If the tile with current index was visible this frame...
    if (tileID_to_frameID_buffer[launch_index] == frameID)
	{
		// Increase the unique tile counter at index 0 with an atomic operation.
        // Note that atomic add returns value found at given memory location BEFORE addition.
		uint unique_tileID_buffer_index = 1u + atomicAdd(&(unique_tileID_buffer[0u].x), 1u);
		size_t unique_tileID_buffer_size = unique_tileID_buffer.size();

		//  Write tileID to unique_tileID_buffer (if any room left).
		if (unique_tileID_buffer_index < make_int1(unique_tileID_buffer_size).x)
		{
			// Translate tile index back to tileID.
			uint mipID, x, y;
            tileIDForOffset(launch_index, mipID, x, y);
			uint3 tileID = make_uint3(mipID, x, y);

			// And store the tileID in the buffer.
			unique_tileID_buffer[unique_tileID_buffer_index] = tileID;
		}
	}
}
