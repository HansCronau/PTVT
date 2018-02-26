#include <algorithm> // std::sort
#include <iostream> // debugging
#include <iterator>
#include <math.h> // pow
#include "vtManager.h"
#include "vtHelpers.h"


/*
 *  Constructor and Destroyer
 */

TileInfo::TileInfo() : mipID(0), x(0), y(0), parent(nullptr), last_frame_id_visible(-999)/*, nrOfReferencesDuringLastFrame(0)*/
{
    for (int i = 0; i < 4; i++)
    {
        children[i] = nullptr;
    }
}

vtManager::vtManager(
	vtTileReader * const tile_reader,
	vtTileCache * const tile_pool,
	vtPageTable * const page_table,
	int const &virtual_texture_dimensions_in_tiles,
	int const &tile_dimensions_in_texels,
	unsigned int const &tile_border_width_in_texels,
	unsigned int const &min_mipID,
	int const &keep_alive_time,
	int const &max_tile_uploads_per_frame
) :
	m_tile_reader(tile_reader),
	m_tile_pool(tile_pool),
	m_page_table(page_table),
	m_virtual_texture_dimensions_in_tiles(virtual_texture_dimensions_in_tiles),
	m_tile_dimensions_in_texels(tile_dimensions_in_texels),
	m_tile_border_width_in_texels(tile_border_width_in_texels),
	m_min_mipID(min_mipID),
	m_keep_alive_time(keep_alive_time),
	m_max_tile_uploads_per_frame(max_tile_uploads_per_frame)
{
    // IDEA: Could assert power of 2 dimensions here for virtual texutre, tile pool, and tiles.

    m_tile_pool_dimensions_in_tiles = tile_pool->DimensionsInTiles();
    m_number_of_mip_levels = (int)log2f((float)m_virtual_texture_dimensions_in_tiles) + 1;
    m_number_of_tiles = offsetForMipID(m_number_of_mip_levels);

    InitialiseTileInfos();

    // Make two arrays containing all tile pool addresses.
	// First array is to keep track of all addresses, second will list the ones that are open.
    m_tile_pool_addresses = new TilePoolAddress*[m_tile_pool_dimensions_in_tiles*m_tile_pool_dimensions_in_tiles];
	m_open_tile_pool_addresses = new TilePoolAddress*[m_tile_pool_dimensions_in_tiles*m_tile_pool_dimensions_in_tiles];
    for (int y = 0; y < m_tile_pool_dimensions_in_tiles; y++)
    {
        for (int x = 0; x < m_tile_pool_dimensions_in_tiles; x++)
        {
			size_t index = y*m_tile_pool_dimensions_in_tiles + x;
			m_tile_pool_addresses[index] = new TilePoolAddress(x, y);
			m_open_tile_pool_addresses[index] = m_tile_pool_addresses[index];
        }
	}

	// Initially all tile pool addresses are open.
	m_number_of_open_tile_pool_addresses = m_tile_pool_dimensions_in_tiles*m_tile_pool_dimensions_in_tiles;

    m_tiles_in_tile_pool = new std::list<TileInfo*>[m_number_of_mip_levels];
    for (unsigned int i = 0; i < m_number_of_mip_levels; i++)
    {
        m_tiles_in_tile_pool[i] = std::list<TileInfo*>();
    }

    m_visible_tiles = new std::vector<TileInfo*>[m_number_of_mip_levels];
    for (unsigned int i = 0; i < m_number_of_mip_levels; i++)
    {
        m_visible_tiles[i] = std::vector<TileInfo*>();
    }

    m_nr_visible_tiles_per_tile_mipID = new int[m_number_of_mip_levels];
    m_nr_newly_visible_tiles_per_tile_mipID = new int[m_number_of_mip_levels];
    m_tiles_to_upload = std::vector<TileInfo*>();
    m_visible_tiles_to_upload_per_mipID = new std::forward_list<TileInfo*>[m_number_of_mip_levels];
    m_page_table_buffer_mipmaps = new optix::float4*[m_number_of_mip_levels];

    m_internal_frame_counter = 0;

    AddMissingMinMipTilesToUploadQueue(0, false);
    UploadTilesToUpload(false);
    UpdatePageTable();

    //PrintDebug("Exiting TileTree constructor");
}

vtManager::~vtManager()
{
    for (unsigned int i = 0; i < m_number_of_tiles; i++)
    {
        delete m_tile_infos[i];
    }
    delete[] m_tile_infos;

    for (int i = 0; i < m_tile_pool_dimensions_in_tiles*m_tile_pool_dimensions_in_tiles; i++)
    {
        delete m_tile_pool_addresses[i];
    }
    delete[] m_tile_pool_addresses;
	delete[] m_open_tile_pool_addresses;

    delete[] m_tiles_in_tile_pool;
    delete[] m_visible_tiles;
    delete[] m_nr_visible_tiles_per_tile_mipID;
    delete[] m_nr_newly_visible_tiles_per_tile_mipID;
    delete[] m_visible_tiles_to_upload_per_mipID;
    delete[] m_page_table_buffer_mipmaps;
}


/*
 *  Getters and Setters
 */

int vtManager::GetNrMipLevels() { return m_number_of_mip_levels; }
int vtManager::GetNrTilesTotal() { return m_number_of_tiles; }
int vtManager::GetKeepAliveTime() { return m_keep_alive_time; }
void vtManager::SetKeepAliveTime(int nr_of_frames) { m_keep_alive_time = nr_of_frames; }
int vtManager::GetNrTilesAlive()
{
    int nrTilesAlive = 0;
    for (int i = 0; i < m_number_of_tiles; i++)
    {
        nrTilesAlive += IsAlive(m_tile_infos[i]);
    }
    return nrTilesAlive;
}
int vtManager::GetNrTilesAlive(int mipID)
{
    const int i_min = offsetForMipID(mipID);
    const int mipDimensions = dimensionsForMipID(mipID);
    const int i_exlcmax = i_min + mipDimensions*mipDimensions;
    int nrTilesAlive = 0;
    for (int i = i_min; i < i_exlcmax; i++)
    {
        nrTilesAlive += IsAlive(m_tile_infos[i]);
    }
    return nrTilesAlive;
}
int vtManager::GetNrTilesVisible() { return m_nr_visible_tiles; }
int vtManager::GetNrTilesVisible(int mipID) { return m_nr_visible_tiles_per_tile_mipID[mipID]; }
int vtManager::GetNrTilesBecameAlive() { return m_nr_newly_visible_tiles; }
int vtManager::GetNrTilesBecameAlive(int mipID) { return m_nr_newly_visible_tiles_per_tile_mipID[mipID]; }

TileInfo* vtManager::GetTileInfo(int mipID, int x, int y)
{
    //PrintDebug("Entering GetTileInfo");
    int tileIndex = offsetForMipID(mipID) + dimensionsForMipID(mipID) * y + x;
    //PrintDebug("Exiting GetTileInfo");
    return m_tile_infos[tileIndex];
}

void vtManager::SetMinMipID(unsigned int min_mipID)
{
    m_min_mipID = min_mipID;
    m_min_mip_up_to_date = false;
}


/*
*  Instruction Methods
*/

void vtManager::PrepareForNewFrame()
{
    //PrintDebug("Entering PrepareForNewFrame");

    //std::cout << "Here we go for " << m_number_of_tiles << " tiles..." << std::endl;

    //for (int i = 0; i < m_number_of_tiles; i++)
    //{
    //	// do something to all tiles
    //}

    m_internal_frame_counter++;

    m_nr_visible_tiles = 0;
    m_nr_newly_visible_tiles = 0;
    for (unsigned int mipID = 0; mipID < m_number_of_mip_levels; mipID++)
    {
        m_visible_tiles[mipID].clear();
        m_nr_visible_tiles_per_tile_mipID[mipID] = 0;
        m_nr_newly_visible_tiles_per_tile_mipID[mipID] = 0;
    }

    //PrintDebug("Exiting PrepareForNewFrame");
}

void vtManager::RegisterTileVisible(int mipID, int x, int y)
{
    if (mipID > static_cast<int>(m_number_of_mip_levels - 1))
    {
        std::cout << "TileTree: Something went terribly wrong. I received a mipID higher than max mipID. MipID should be capped on GPU side." << std::endl;
        exit(16);
    }

    //PrintDebug("Entering RegisterTileVisible");
    TileInfo* tileInfo = GetTileInfo(mipID, x, y);

	// Loop over tile and parents to set all visible.
	while (tileInfo != nullptr)
	{
		if (tileInfo->last_frame_id_visible == m_internal_frame_counter)
		{
			// Tile was already registered this frame. Same must be true for all parents. Break loop.
			break;
		}
		else
		{
			// This tile hasn't been registered during this frame before.

			// Register some statistics.
			m_nr_visible_tiles++;
            m_nr_visible_tiles_per_tile_mipID[tileInfo->mipID]++;
			if (tileInfo->last_frame_id_visible + 1 != m_internal_frame_counter)
			{
				// This tile wasn't visible during last frame.
				// m_nr_tiles_became_visible_since_last_frame++;
				// m_nr_tiles_became_visible_since_last_frame_per_tile_mipID[mipID]++;
			}
			if (!IsAlive(tileInfo))
			{
				// This tile hasn't been visible for over m_keep_alive_time nr of frames.
                m_nr_newly_visible_tiles++;
                m_nr_newly_visible_tiles_per_tile_mipID[tileInfo->mipID]++;
				// m_nr_tiles_became_visible_minus_keep_alive++;
				// m_nr_tiles_became_visible_minus_keep_alive_per_tile_mipID[mipID]++;
			}
			if (!tileInfo->isInTilePool)
			{
				// This tile isn't stored in the tile pool.
				// m_nr_tiles_visible_not_in_pool++;
				// m_nr_tiles_visible_not_in_pool_per_tile_mipID[mipID]++;
			}
			
			// Set current frameID as the last time the tile was visible.
			tileInfo->last_frame_id_visible = m_internal_frame_counter; 
			
			// List it as visible.
			m_visible_tiles[tileInfo->mipID].push_back(tileInfo);

			// Loop over parents, because they are inherently visible as well.
			tileInfo = tileInfo->parent;
		}
	}

    //PrintDebug("Exiting RegisterTileVisible");
}

void vtManager::UpdateVirtualTexture()
{
    ClearTilesToUpload();
    unsigned int firstMissingTileIndex;
    if (!IsMinMipUpToDate(firstMissingTileIndex))
    {
		// The program will never get here if all required tiles are force loaded during init. However, one may choose to only load tile 0 0 0 at that point.
        AddMissingMinMipTilesToUploadQueue(firstMissingTileIndex);
    }
    AddVisibleTilesToUploadQueue(true); // Set to false if all visible tiles should be loaded at once. (Will cause framedrop.)
    UploadTilesToUpload(true);
    UpdatePageTable();
}


/*
*  Debugging
*/

void vtManager::PrintDebug(std::string s)
{
    //return;
    std::cout << s << std::endl;
    std::cout << "TileTree contains:" << std::endl;
    std::cout << "  Dimensions virtual texture in tiles: " << m_virtual_texture_dimensions_in_tiles << std::endl;
    std::cout << "  Nr mipmap levels: " << m_number_of_mip_levels << std::endl;
    std::cout << "  Total nr of tiles: " << m_number_of_tiles << std::endl;
    std::cout << "  MipID first tile: " << m_tile_infos[0]->mipID << std::endl;
    std::cout << "  MipID indices:" << std::endl;

    for (unsigned int i = 0; i < m_number_of_mip_levels; i++)
    {
        std::cout << "    " << offsetForMipID(i) << std::endl;
    }

    size_t totalNrVisibleTiles = 0;
    for (unsigned int i = 0; i < m_number_of_mip_levels; i++)
    {
        totalNrVisibleTiles += m_visible_tiles[i].size();
    }
    std::cout << "  Visible tiles this frame: " << totalNrVisibleTiles << std::endl;

    size_t totalNrOfTilesInTilePool = 0;
    for (unsigned int i = 0; i < m_number_of_mip_levels; i++)
    {
        totalNrOfTilesInTilePool += m_tiles_in_tile_pool[i].size();
    }
    std::cout << "  Tiles in tile pool this frame: " << totalNrOfTilesInTilePool << std::endl;
}


/*
*  Private Methods
*/

// Initialises TileInfo objects and builds an array and tree of pointers to an between them.
// Order in the array is low to high mipID, low to high y, low to high x.
void vtManager::InitialiseTileInfos()
{
    //PrintDebug("Entering InitialiseTileInfos");

    m_tile_infos = new TileInfo*[m_number_of_tiles];
    for (unsigned int i = 0; i < m_number_of_tiles; i++)
    {
        m_tile_infos[i] = new TileInfo();
    }

    for (unsigned int mipID = 0; mipID < m_number_of_mip_levels; mipID++)
    {
        for (int y = 0; y < dimensionsForMipID(mipID); y++)
        {
            for (int x = 0; x < dimensionsForMipID(mipID); x++)
            {
                TileInfo* tile = GetTileInfo(mipID, x, y);
                tile->mipID = mipID;
                tile->x = x;
                tile->y = y;
                tile->parent = mipID == 0 ? nullptr : GetTileInfo(mipID - 1, x / 2, y / 2);
                if (mipID == m_number_of_mip_levels - 1)
                {
                    tile->children[0] = nullptr;
                    tile->children[1] = nullptr;
                    tile->children[2] = nullptr;
                    tile->children[3] = nullptr;
                }
                else
                {
                    tile->children[0] = GetTileInfo(mipID + 1, x * 2, y * 2);
                    tile->children[1] = GetTileInfo(mipID + 1, x * 2, y * 2 + 1);
                    tile->children[2] = GetTileInfo(mipID + 1, x * 2 + 1, y * 2);
                    tile->children[3] = GetTileInfo(mipID + 1, x * 2 + 1, y * 2 + 1);
                }
                tile->last_frame_id_visible = -m_keep_alive_time;
                tile->isInTilePool = false;
            }
        }
    }
    // std::cout << m_number_of_tiles << "," << counter << std::endl; // debug check, check, double check
    //std::cout << "Number of tiles: " << m_number_of_tiles << "." << std::endl; // debug check, check, double check
    //PrintDebug("Exiting InitialiseTiles");
}

void vtManager::ClearTilesToUpload()
{
    // Prepare an empty list for TileInfos of tiles to be uploaded this frame.
    while (!m_tiles_to_upload.empty())
    {
        m_tiles_to_upload.back()->isInUploadQueue = false;
        m_tiles_to_upload.pop_back();
    }
    for (unsigned int mipID = 0; mipID < m_number_of_mip_levels; mipID++)
    {
		m_visible_tiles_to_upload_per_mipID[mipID].clear();
    }
}

// Loops over tiles infos, low mipID to high, checking if all up to m_min_mipID are present in the tile pool. If not: adds tile to this frame's upload queue.
void vtManager::AddMissingMinMipTilesToUploadQueue(unsigned int const startTileIndex, bool const respect_max_tile_upload)
{
    unsigned int lastIndexToUpload = offsetForMipID(m_min_mipID + 1) - 1;
    if (lastIndexToUpload > m_number_of_tiles)
    {
        lastIndexToUpload = m_number_of_tiles;
    }

    for (unsigned int tileIndex = startTileIndex; tileIndex <= lastIndexToUpload && (!respect_max_tile_upload || m_tiles_to_upload.size() <= m_max_tile_uploads_per_frame); tileIndex++)
    {
		TileInfo* tile = m_tile_infos[tileIndex];
        if (!tile->isInTilePool)
        {
			// Find tile pool address.
			bool is_tilepool_saturated, was_assigned_address;
			AssignTilePoolAddress(tile, is_tilepool_saturated, was_assigned_address);

			if (was_assigned_address)
			{
				// Move tile to upload queue.
				tile->isInUploadQueue = true;
				m_tiles_to_upload.push_back(tile);
			}
			else
			{
				// All following tiles in m_tile_infos will have same or higher mipID. No need to continue trying to find an address.
				std::cout << "Warning: Could not upload all tiles up to minimum mipmap level, because tile pool is full." << std::endl;
				break;
			}
        }
    }
}

void vtManager::AddVisibleTilesToUploadQueue(bool respect_max_tile_upload)
{
    // Of tiles visible this frame, select the most important ones not yet in the tile pool.
	// Loop over mipIDs, lower mipIDs first.
	// Limit to a maximum nr of uploads per frame.
    for (unsigned int mipID = 0; mipID < m_number_of_mip_levels && (!respect_max_tile_upload || m_tiles_to_upload.size() <= m_max_tile_uploads_per_frame); mipID++)
    {
        for (unsigned int i = 0; i < m_visible_tiles[mipID].size() && (!respect_max_tile_upload || m_tiles_to_upload.size() <= m_max_tile_uploads_per_frame); i++)
        {
			// Get tile from visible tiles.
            TileInfo* tileToAdd = m_visible_tiles[mipID][i];

            if ( !tileToAdd->isInTilePool && !tileToAdd->isInUploadQueue )
            {
				// Add tile to upload queue.
                m_visible_tiles_to_upload_per_mipID[tileToAdd->mipID].push_front(tileToAdd);
            }
        }
    }

	// Find a tile pool address for tiles in m_visible_tiles_to_upload_per_mipID.
	// If pool address found, add tile to tiles to m_tiles_to_upload, respecting m_max_tile_uploads_per_frame if respect_max_tile_upload.
	// Loop over mipIDs, low to high. Assume tiles up to m_min_mipID already in tile pool.
	bool is_tilepool_saturated = false;
	bool was_assigned_address;
    for (unsigned int mipID = m_min_mipID + 1; mipID < m_number_of_mip_levels; mipID++)
    {
        while (!m_visible_tiles_to_upload_per_mipID[mipID].empty() && (!respect_max_tile_upload || m_tiles_to_upload.size() <= m_max_tile_uploads_per_frame))
        {
			// Find tile pool address.
			TileInfo* tile = m_visible_tiles_to_upload_per_mipID[mipID].front();
			AssignTilePoolAddress(tile, is_tilepool_saturated, was_assigned_address);

			if (was_assigned_address)
			{
				// Move tile to upload queue.
				tile->isInUploadQueue = true;
				m_tiles_to_upload.push_back(tile);
				m_visible_tiles_to_upload_per_mipID[mipID].pop_front();
			}
			else
			{
				// All following tiles in m_visible_tiles_to_upload_per_mipID will have same or higher mipID. No need to continue trying to find an address.
				break;
			}
        }
    }
	
	if (is_tilepool_saturated)
	{
		std::cout << "Warning: Tile Pool saturation. " << (was_assigned_address ? "Trashing higher res tiles." : "No tiles higher res left to trash.") << " Adjust mipmap bias." << std::endl;
		AdjustMipmapBias();
	}
}

void vtManager::UploadTilesToUpload(bool respect_max_tile_upload)
{
    // Upload tiles to upload
	for (int i = 0; i < m_tiles_to_upload.size() && (!respect_max_tile_upload || i <= m_max_tile_uploads_per_frame); i++)
    {
		MapTile(m_tiles_to_upload[i]);
		ilImage tile;
		m_tile_reader->GetTileImage(tile, m_tiles_to_upload[i]->mipID, m_tiles_to_upload[i]->x, m_tiles_to_upload[i]->y);
		m_tile_pool->UploadTile(
			tile,
			m_tiles_to_upload[i]->tilePoolAddress->x,
			m_tiles_to_upload[i]->tilePoolAddress->y
		);
    }
}

// Checks if all tiles up to the minimum required mipmap level are present in the tile pool.
bool vtManager::IsMinMipUpToDate(unsigned int& firstMissingTileIndex)
{
    if (m_min_mip_up_to_date)
    {
        return true;
    }

    unsigned int lastIndexToCheck = offsetForMipID(m_min_mipID + 1) - 1;
    if (lastIndexToCheck > m_number_of_tiles)
    {
        lastIndexToCheck = m_number_of_tiles;
    }

    for (unsigned int tileIndex = 0; tileIndex < lastIndexToCheck; tileIndex++)
    {
        if (!m_tile_infos[tileIndex]->isInTilePool)
        {
            firstMissingTileIndex = tileIndex;
            return false;
        }
    }

    m_min_mip_up_to_date = true;
    return true;
}

bool vtManager::IsAlive(TileInfo* tileInfo)
{
    return tileInfo->last_frame_id_visible + m_keep_alive_time >= m_internal_frame_counter;
}

//void TileTree::PrioritiseUploadQueue()
//{
//    std::sort(m_tiles_to_upload.begin(), m_tiles_to_upload.end(), TileShouldGoBeforeTile);
//}
//
//bool TileTree::TileShouldGoBeforeTile(TileInfo* tileA, TileInfo* tileB)
//{
//    return tileA->mipID < tileB->mipID;
//}

// Returns false if tilepool is saturated.
void vtManager::AssignTilePoolAddress(TileInfo* tile, bool& is_tilepool_saturated, bool& was_assigned_address)
{
	is_tilepool_saturated = false; // Assume tilepool is not (yet) saturated.


	tile->tilePoolAddress = GetFreeTilePoolAddress();
	is_tilepool_saturated = tile->tilePoolAddress == nullptr;
	if (is_tilepool_saturated)
	{
		// There were no unclaimed tile pool addresses available. The tile pool is saturated.

		tile->tilePoolAddress = GetNonFreeTilePoolAddress(tile->mipID);
		was_assigned_address = tile->tilePoolAddress != nullptr;
		// If not was_assigned_address that means all tiles in the tile pool are of the same or lower mipID than given tile.
	}
}

// Returns a free TilePoolAddress. No tiles are mapped to this address. Returns NULL if no addresses are free.
TilePoolAddress* vtManager::GetFreeTilePoolAddress()
{
    // First check if there are open spots in the tile pool.
    if (m_number_of_open_tile_pool_addresses > 0)
    {
        return m_open_tile_pool_addresses[--m_number_of_open_tile_pool_addresses];
    }

    // If there are no open spots, check which tiles in the pool have been invisible for longer than their keep-alive-time. Start with the highest resolution tiles.
    for (unsigned int mipID = m_number_of_mip_levels - 1; mipID != -1; mipID--) // Reverse loop over mip levels. // TODO(HansCronau): Condition mipID > 0, was replaced by mipID != -1 to find more bugs. Revert.
    {
        for (std::list<TileInfo*>::iterator i = m_tiles_in_tile_pool[mipID].begin(); i != m_tiles_in_tile_pool[mipID].end(); i++)
        {
            if (!IsAlive(*i))
            {
                // This tile hasn't been visible for over m_keep_alive_time nr of frames. It is a candidate for removal.
                UnmapTile(*i);
                // After UnmapTile() we know for sure there is an open spot in the tile pool. No need to check if there are.
                return m_open_tile_pool_addresses[--m_number_of_open_tile_pool_addresses];
            }
            else
            {
                continue;
            }
        }
    }

    // If we have come here the tile pool is saturated (all tiles in it have been visible within the last m_keep_alive_time frames) and we cannot return a meaningful address.
    return nullptr;
}

// Returns a non-free TilePoolAddress by unmapping a high res tile if available.
// Unmapped tile is as high res as possible and must have mipID higher than the mipID to be mapped
// Returns NULL if no tile could be unmapped.
TilePoolAddress* vtManager::GetNonFreeTilePoolAddress(unsigned int forMipID)
{
	// Loop over all tiles in the tile pool, starting with the highest resolution tiles.
	for (unsigned int mipID = m_number_of_mip_levels - 1; mipID > forMipID && mipID != -1; mipID--) // Reverse loop over mip levels.
	{
		for (std::list<TileInfo*>::iterator i = m_tiles_in_tile_pool[mipID].begin(); i != m_tiles_in_tile_pool[mipID].end(); i++)
		{
			// Can assume all tiles were visible within keep-alive-time. Don't loop over all tiles in mip-level. Too much work.
			UnmapTile(*i);
			return m_open_tile_pool_addresses[--m_number_of_open_tile_pool_addresses];
		}
	}

	// If we have come here the tile pool does not contain tiles of a mipID higher than the one that needs to be mapped.
	return nullptr;
}

// Note: Two steps: give up tilePoolAddress. Then unmap from being in tile pool.
void vtManager::UnmapTile(TileInfo* tileInfo)
{
    m_open_tile_pool_addresses[m_number_of_open_tile_pool_addresses] = tileInfo->tilePoolAddress;
    m_number_of_open_tile_pool_addresses++;
    tileInfo->tilePoolAddress = nullptr;
    tileInfo->isInTilePool = false;
    m_tiles_in_tile_pool[tileInfo->mipID].erase(tileInfo->tiles_in_tile_pool_iterator);
}

void vtManager::AdjustMipmapBias()
{
	// TODO(HansCronau): An industry applied virtual texture system should adjust mipmap bias when it detects tile pool saturation.
}

// Note: Assumes tile already has a tilePoolAddress reserved. (This is because of the delay between determination and having loaded the tile.) Only maps it to actually be in the tile pool.
void vtManager::MapTile(TileInfo* tileInfo)
{
	tileInfo->isInTilePool = true;
    m_tiles_in_tile_pool[tileInfo->mipID].push_back(tileInfo);
    tileInfo->tiles_in_tile_pool_iterator = --m_tiles_in_tile_pool[tileInfo->mipID].end();
}

// Assumes mipID 0 is always in tile pool!
// Currently loops over ALL tile infos. Could be optimised by only updating above minMipID and by keeping track of changed tiles uploaded since last frame and only updating those and their descendants.
void vtManager::UpdatePageTable()
{
    // Map all page table buffer mipmaps.
    for (unsigned int mipID = 0; mipID < m_number_of_mip_levels; mipID++)
    {
        m_page_table_buffer_mipmaps[mipID] = static_cast<optix::float4*>(m_page_table->Buffer()->map(mipIDToMipLevel(mipID, m_virtual_texture_dimensions_in_tiles)));
    }
    
    // Small debug check. Program would crash if this were true.
    if (!m_tile_infos[0]->isInTilePool)
    {
        std::cout << "\nFatal error: MipID 0 is not residing in tile pool. Virtual texture cannot fall back.\n" << std::endl;
        exit(17);
    }

    // Loop over all tile infos, from low mipmap to high mipmap, adding their address details to the page table.
    // Currently loop uses no prior knowledge about m_number_of_tiles array. Could be optimised by calculating mipDimensions per mip level.


    TileInfo* tile;
    for (unsigned int mipID = 0; mipID < m_number_of_mip_levels; mipID++)
    {
        unsigned int mipDimensions = dimensionsForMipID(mipID);
        for (unsigned int y = 0; y < mipDimensions; y++)
        {
            for (unsigned int x = 0; x < mipDimensions; x++)
            {
                tile = GetTileInfo(mipID, x, y);

                // Find the first resident ancestor. Can be tile itself. Assumes that at least one ancestor is resident in tile pool
                // If tile is in pool it itself is first resident ancestor.
                if (tile->isInTilePool)
                {
                    tile->firstResidentAnchestor = tile;
                }
                // Otherwise, look for least removed resident ancestor, a pointer to which is stored by parent.
                else
                {
                    tile->firstResidentAnchestor = tile->parent->firstResidentAnchestor;
                }
                optix::float4 addressData = TileInfoToScaleBias(tile);
                m_page_table_buffer_mipmaps[tile->mipID][mipDimensions * tile->y + tile->x] = addressData;
            }
        }
    }

    // Unmap all mip levels.
    for (unsigned int mipID = 0; mipID < m_number_of_mip_levels; mipID++)
    {
        m_page_table->Buffer()->unmap(mipIDToMipLevel(mipID, m_virtual_texture_dimensions_in_tiles));
    }
}

optix::float4 vtManager::TileInfoToScaleBias(TileInfo* tileInfo)
{
    // For fallbacks, calculate the difference between desired mipmap and fallback mipmap. (0 if actual desired tile is used instead of fallback.)
	const unsigned int fallback_mip_difference = tileInfo->mipID - tileInfo->firstResidentAnchestor->mipID;
    
	// Calcualte scale difference between virtual mipmap level and physical mipmap level. (1.0f if actual desired tile is used instead of fallback.)
	const float fallback_scale = 1.0f / static_cast<float>(1 << fallback_mip_difference);

	// Fallbacks will have a within-tile offset since not the whole physical tile is used but only a section of it.
	// Determine within-tile offset of this section(expressed in tiles) by scaling and taking fractional part.
	float dummy;
	const float within_tile_offset_x = modf(static_cast<float>(tileInfo->x) * fallback_scale, &dummy);
	const float within_tile_offset_y = modf(static_cast<float>(tileInfo->y) * fallback_scale, &dummy);
    
    // Add within-tile offset to physical tile address. Fallbacks will give nonintegral tile coordinates.
	const float physical_tile_x = (tileInfo->firstResidentAnchestor->tilePoolAddress->x + within_tile_offset_x);
	const float physical_tile_y = (tileInfo->firstResidentAnchestor->tilePoolAddress->y + within_tile_offset_y);

    // Calculate scale and bias based on collected information.
	return create_scale_and_bias(
		tileInfo->mipID,
		tileInfo->x,
		tileInfo->y,
		fallback_scale,
		m_tile_dimensions_in_texels,
		m_tile_border_width_in_texels,
		m_tile_pool_dimensions_in_tiles * m_tile_dimensions_in_texels,
		physical_tile_x, physical_tile_y
	);
}

std::vector<TileInfo*>* vtManager::GetVisibleTilesInfo()
{
    return m_visible_tiles;
}
