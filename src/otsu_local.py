def local_otsu_binarize(img: np.ndarray, tile_size: int = 64) -> np.ndarray:
    """Wendet Otsu-Thresholding lokal auf ein Bild an, aufgeteilt in Kacheln (Tiles)."""
    from skimage.filters import threshold_otsu

    h, w = img.shape
    binary = np.zeros_like(img, dtype=bool)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]
            if tile.size == 0:
                continue
            t = threshold_otsu(tile)
            binary[y:y+tile_size, x:x+tile_size] = tile > t

    return binary
