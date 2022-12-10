import argparse


def get_args():
    """processes input arguments
    Parameters
    ----------
    None

    Returns
    -------
    parser.parse_args() - arguments to be used in script
    """

    parser = argparse.ArgumentParser()

    # add argument for file name
    parser.add_argument('--file_name',
                        type=str,
                        help='name of the image file',
                        required=True)

    # add argument for min_distance
    parser.add_argument('--min_distance',
                        type=int,
                        help='minimum pixel distance between maxima',
                        default=10)

    # add argument for min_thresh_maxima
    parser.add_argument('--min_thresh_maxima',
                        type=float,
                        help='minimum relative intensity threshold for maxima',
                        default=0.8)

    # add argument for min_thresh_contours
    parser.add_argument('--min_thresh_contours',
                        type=float,
                        help='minimum relative threshold for contours',
                        default=0.8)

    # add argument for thresh_step
    parser.add_argument('--thresh_step',
                        type=int,
                        help='step size for finding contours',
                        default=10)

    # ADD OPTIONS TO NOT INCLUDE init/sim/out_filter
    parser.add_argument('--no_init_filter',
                        action='store_true')

    parser.add_argument('--no_sim_filter',
                        action='store_true')

    parser.add_argument('--no_out_filter',
                        action='store_true')

    return parser.parse_args()
