import pandas as pd
import os


def create_measurements_csv(output_path, data):
    """
    Create a new measurements.csv with the correct format for ProcessNewSessionData.py.

    Args:
        output_path (str): Path to save the new measurements.csv
        data (list): List of lists containing CSV data
    """
    try:
        # Define columns
        columns = ['filename', 'x', 'y', 'z', 'r', 'distance_mm', 'angle_deg',
                   'first_corner_x_y', 'last_corner_x_y', 'timestamp']

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Save as comma-delimited CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, sep=',')
        print(f"Generated CSV saved to {output_path}")

        # Print generated DataFrame
        print("Generated DataFrame:")
        print(df.to_string(index=False))
        print("Columns:", list(df.columns))

        return df

    except Exception as e:
        print(f"Error generating CSV: {e}")
        return None


def main():
    output_path = "../calibration_images/session_20250517_050110/measurements.csv"

    # Hardcoded data (21 rows, 10 columns)
    data = [
        ['../calibration_images/session_20250517_050110/image001.jpg', 300.001739, -0.003839, 0.001056, -0.00348, 50.0,
         0.08, '918.0649_597.6677', '1368.3062_1001.64374', '20250517_050122'],
        ['../calibration_images/session_20250517_050110/image002.jpg', 260.984475, -29.210269, -0.014499, -0.082755,
         50.0, 174.18, '993.26_487.56', '482.78_183.41', '20250517_050124'],
        ['../calibration_images/session_20250517_050110/image003.jpg', 279.096405, -30.02123, 0.003644, 0.032109, 50.0,
         173.48, '996.37_596.54', '471.55_282.75', '20250517_050126'],
        ['../calibration_images/session_20250517_050110/image004.jpg', 299.153517, -30.022555, -0.010878, 0.036901,
         50.0, 174.29, '1009.15_717.36', '486.56_400.35', '20250517_050128'],
        ['../calibration_images/session_20250517_050110/image005.jpg', 319.1809, -30.023667, 0.000236, 0.0096, 50.0,
         175.04, '1020.3_837.79', '499.93_518.94', '20250517_050130'],
        ['../calibration_images/session_20250517_050110/image006.jpg', 339.082801, -30.027627, 0.012349, -0.017927,
         50.0, 175.83, '1029.95_959.21', '511.58_639.06', '20250517_050131'],
        ['../calibration_images/session_20250517_050110/image007.jpg', 260.947253, -10.25345, -0.018365, 0.023993, 50.0,
         176.32, '1124.96_494.78', '615.56_169.08', '20250517_050134'],
        ['../calibration_images/session_20250517_050110/image008.jpg', 279.168513, -10.003584, 0.014398, -0.00328, 50.0,
         177.85, '1134.13_595.61', '634.01_244.82', '20250517_050135'],
        ['../calibration_images/session_20250517_050110/image009.jpg', 299.009268, -10.004926, -0.015061, 0.000698,
         50.0, 178.13, '1137.82_716.73', '638.54_364.26', '20250517_050137'],
        ['../calibration_images/session_20250517_050110/image010.jpg', 319.014073, -10.005165, 0.009901, -0.000101,
         50.0, 178.41, '1140.82_838.09', '642.59_487.64', '20250517_050139'],
        ['../calibration_images/session_20250517_050110/image011.jpg', 339.003586, -10.004704, -0.003536, 0.00696, 50.0,
         178.71, '1143.31_959.48', '646.15_610.12', '20250517_050141'],
        ['../calibration_images/session_20250517_050110/image012.jpg', 260.945204, 9.742595, -0.017673, 0.006842, 50.0,
         179.5, '1203.01_721.26', '704.55_395.29', '20250517_050143'],
        ['../calibration_images/session_20250517_050110/image013.jpg', 279.080626, 9.99507, 0.015305, -0.022539, 50.0,
         1.93, '804.79_140.86', '1275.16_553.78', '20250517_050144'],
        ['../calibration_images/session_20250517_050110/image014.jpg', 299.01115, 9.991583, -0.014999, -0.01425, 50.0,
         2.01, '793.55_345.6', '1266.55_732.55', '20250517_050146'],
        ['../calibration_images/session_20250517_050110/image015.jpg', 319.09935, 9.992513, 0.00894, -0.013635, 50.0,
         1.81, '787.46_471.55', '1261.51_853.33', '20250517_050148'],
        ['../calibration_images/session_20250517_050110/image016.jpg', 339.095152, 9.991121, -0.005811, -0.012456, 50.0,
         1.62, '782.33_595.83', '1256.75_973.42', '20250517_050150'],
        ['../calibration_images/session_20250517_050110/image017.jpg', 260.994665, 29.725499, -0.021161, 0.004677, 50.0,
         1.79, '825.86_468.04', '1306.41_815.37', '20250517_050152'],
        ['../calibration_images/session_20250517_050110/image018.jpg', 279.00427, 30.008611, 0.004696, -0.032659, 50.0,
         6.56, '964.37_232.73', '1406.74_648.83', '20250517_050154'],
        ['../calibration_images/session_20250517_050110/image019.jpg', 299.075954, 30.010907, -0.01535, -0.043128, 50.0,
         5.86, '947.54_351.35', '1392.58_765.74', '20250517_050156'],
        ['../calibration_images/session_20250517_050110/image020.jpg', 319.177205, 30.007869, 0.000775, -0.045309, 50.0,
         5.16, '931.53_475.36', '1379.67_883.23', '20250517_050158'],
        ['../calibration_images/session_20250517_050110/image021.jpg', 339.164811, 30.008648, 0.012068, -0.011191, 50.0,
         4.51, '918.02_596.59', '1368.36_1001.17', '20250517_050159']
    ]

    # Create CSV
    generated_df = create_measurements_csv(output_path, data)

    if generated_df is not None:
        print("CSV generation completed successfully!")
    else:
        print("CSV generation failed.")


if __name__ == "__main__":
    main()