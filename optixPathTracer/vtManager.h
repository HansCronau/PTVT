#include <vector>
#include <list>
#include <forward_list>
#include <iterator>
#include <string> // debug
#include "vtTileReader.h"
#include "vtTileCache.h"
#include "vtPageTable.h"

struct TileID
{
	TileID(int _mipID, int _x, int _y) : mipID(_mipID), x(_x), y(_y) {}
	unsigned int mipID, x, y;
};

struct TilePoolAddress
{
	TilePoolAddress(int _x, int _y) : x(_x), y(_y) {}
	unsigned int x, y;
};

struct TileInfo
{
	TileInfo();
	int mipID, x, y; // 2^mipID = single dimension nr of tiles in mip-level. (2^mipID)^2 = total nr of tiles in mip-level.
	TileInfo* parent;
	TileInfo* children[4]; // always has 4 children
	TileInfo* firstResidentAnchestor;
	int last_frame_id_visible; // FrameID of last frame when tile was visible
	//int nrOfReferencesDuringLastFrame;
	TilePoolAddress* tilePoolAddress = nullptr;
	bool isInTilePool = false;
	bool isInUploadQueue = false;
	std::list<TileInfo*>::iterator tiles_in_tile_pool_iterator;
	//int GetFramesSinceLastVisible(int currentFameID);
};

class vtManager
{
public:
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
	);
	~vtManager();
	
	// Getters and Setters
	TileInfo* GetTileInfo(int mipID, int x, int y);
	int GetNrTilesTotal();
	int GetNrMipLevels();
	int GetKeepAliveTime();
    void SetKeepAliveTime(int nr_of_frames);
    int GetNrTilesAlive();
    int GetNrTilesAlive(int mipID);
	int GetNrTilesVisible();
    int GetNrTilesVisible(int mipID);
    int GetNrTilesBecameAlive();
    int GetNrTilesBecameAlive(int mipID);
	std::vector<TileInfo*>* GetVisibleTilesInfo();
	void SetMinMipID(unsigned int min_mipID);

	// Instruction Methods
	void PrepareForNewFrame();
	void RegisterTileVisible(int mipID, int x, int y);
	void UpdateVirtualTexture();

	// Debug
	void PrintDebug(std::string s);

private:

	// Private Methods
	void InitialiseTileInfos();
	void ClearTilesToUpload();
	void AddMissingMinMipTilesToUploadQueue(unsigned int startTileIndex = 0, bool respect_max_tile_upload = true);
	void AddVisibleTilesToUploadQueue(bool respect_max_tile_upload = true);
	void UploadTilesToUpload(bool respect_max_tile_upload = true);
    bool IsMinMipUpToDate(unsigned int& firstMissingTileIndex);
    bool IsAlive(TileInfo* tileInfo);
	//void PrioritiseUploadQueue();
	//static bool TileShouldGoBeforeTile(TileInfo* tileA, TileInfo* tileB);
	void vtManager::AssignTilePoolAddress(TileInfo* tile, bool& is_tilepool_saturated, bool& was_assigned_address);
	TilePoolAddress* GetFreeTilePoolAddress();
	TilePoolAddress* GetNonFreeTilePoolAddress(unsigned int forMipID);
	void AdjustMipmapBias();
	void MapTile(TileInfo* tileInfo);
	void UnmapTile(TileInfo* tileInfo);
	void UpdatePageTable();
	optix::float4 TileInfoToScaleBias(TileInfo* tileInfo);

	// Global attributes
	int m_virtual_texture_dimensions_in_tiles;
	unsigned int m_virtual_texture_dimensions_in_texels;
	int m_tile_pool_dimensions_in_tiles;
	unsigned int m_tile_dimensions_in_texels;
	unsigned int m_tile_border_width_in_texels;
	unsigned int m_number_of_mip_levels;
	unsigned int m_number_of_tiles;
	unsigned int m_min_mipID;
	int m_keep_alive_time;
	int m_max_tile_uploads_per_frame;
	TileInfo** m_tile_infos;
	TilePoolAddress** m_tile_pool_addresses;
	TilePoolAddress** m_open_tile_pool_addresses;
	unsigned int m_number_of_open_tile_pool_addresses;
	std::list<TileInfo*>* m_tiles_in_tile_pool;
	vtTileReader* m_tile_reader;
	vtTileCache* m_tile_pool;
	vtPageTable* m_page_table;

	// Per Frame attributes
	int m_internal_frame_counter;
	std::vector<TileInfo*>* m_visible_tiles;
	int m_nr_visible_tiles, m_nr_newly_visible_tiles;
	int *m_nr_visible_tiles_per_tile_mipID, *m_nr_newly_visible_tiles_per_tile_mipID;
	bool m_min_mip_up_to_date;
	std::forward_list<TileInfo*>* m_visible_tiles_to_upload_per_mipID;
	std::vector<TileInfo*> m_tiles_to_upload;
	optix::float4** m_page_table_buffer_mipmaps;
};
