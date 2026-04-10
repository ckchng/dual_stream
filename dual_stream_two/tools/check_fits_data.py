import os
from astropy.io import fits
import numpy as np

def check_fits_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Opening FITS file: {file_path}")
    print("-" * 50)

    try:
        with fits.open(file_path) as hdul:
            # 1. Show summary info
            hdul.info()
            print("\n" + "-" * 50)

            # 2. Iterate through each extension (HDU)
            for i, hdu in enumerate(hdul):
                print(f"\n[HDU {i}]")
                print(f"Name: {hdu.name}")
                print(f"Type: {type(hdu)}")
                
                # Check for image data
                if hdu.is_image:
                    if hdu.data is not None and hdu.data.size > 0:
                        print(f"Data Shape: {hdu.data.shape}")
                        print(f"Data Type: {hdu.data.dtype}")
                        print(f"Min Value: {np.min(hdu.data)}")
                        print(f"Max Value: {np.max(hdu.data)}")
                        print(f"Mean Value: {np.mean(hdu.data):.2f}")
                    else:
                        print(f"Data: {hdu.data.shape if hdu.data is not None else 'None'} (Zero size or Header only)")
                
                # Check for table data
                elif isinstance(hdu, (fits.TableHDU, fits.BinTableHDU)):
                    print(f"Table Rows: {len(hdu.data)}")
                    print(f"Columns: {hdu.columns.names}")

                # 3. Print Header (First 20 cards or all if requested)
                print("\nHeader (truncated):")
                header = hdu.header
                # Print first 20 lines of header or all if it's shorter
                for j, (key, value) in enumerate(header.items()):
                    if j > 25:
                        print("...")
                        break
                    print(f"{key:8} = {value}")

    except Exception as e:
        print(f"Error reading FITS file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fits_path = '/media/ckchng/internal2TB/RAW_DATA/FIREOPAL000/FITS/000_2020-12-08_091608_E_DSC_0001_header.fits'
    check_fits_file(fits_path)
