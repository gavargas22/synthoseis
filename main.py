"""Entry point for building a syntehtic model."""

import argparse
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

from datagenerator.Closures import Closures
from datagenerator.Faults import Faults
from datagenerator.Geomodels import Geomodel
from datagenerator.Horizons import build_unfaulted_depth_maps, create_facies_array
from datagenerator.Parameters import Parameters
from datagenerator.Seismic import SeismicVolume
from datagenerator.RPMConfig import RPMConfig
from datagenerator.util import plot_3D_closure_plot


def build_model(user_json: str, run_id, test_mode: int = None, rpm_factors=None):
    """Build model from config file."""
    # Set up model parameters
    p = Parameters(user_json, runid=run_id, test_mode=test_mode)

    # Build un-faulted depth maps and facies array
    depth_maps, onlap_list, fan_list, fan_thicknesses = build_unfaulted_depth_maps(p)
    facies = create_facies_array(p, depth_maps, onlap_list, fan_list)

    # Build un-faulted geological models
    geo_models = Geomodel(p, depth_maps, onlap_list, facies)

    # Build Faults
    f = Faults(p, depth_maps, onlap_list, geo_models, fan_list, fan_thicknesses)
    # Apply faults to depth maps and age volume, and output faulted files
    f.apply_faulting_to_geomodels_and_depth_maps()
    # Build faulted lithology, net_to_gross and depth and randomised models
    f.build_faulted_property_geomodels(facies)

    # Create closures, remove false closures and and output closures
    closures = Closures(p, f, facies, onlap_list)
    closures.create_closures()
    closures.write_closure_volumes_to_disk()

    # Create 3D qc plot
    if p.qc_plots:
        try:
            plot_3D_closure_plot(p, closures)
        except ValueError:
            p.write_to_logfile("3D Closure Plotting Failed")

    # Create Rho, Vp, Vs volumes, apply Zoeppritz and write seismic volumes to disk
    seismic = SeismicVolume(p, f, closures)
    seismic.build_elastic_properties("inv_vel")
    seismic.build_seismic_volumes()

    closures.write_closure_info_to_log(seismic.rfc_raw[1:4, ...])

    elapsed_time = datetime.datetime.now() - p.start_time
    print("\n\n\n...elapsed time is {}".format(elapsed_time))
    p.write_to_logfile(
        f"elapsed_time: {elapsed_time}\n",
        mainkey="model_parameters",
        subkey="elapsed_time",
        val=elapsed_time,
    )
    p.write_sqldict_to_logfile(f"{p.work_subfolder}/sql_log.txt")
    p.write_sqldict_to_db()

    # Cleanup
    p.h5file.close()
    os.system("rm -rf " + p.temp_folder)
    try:
        os.system(f"chmod -R 777 {p.work_subfolder}")
    except OSError:
        pass

    # Change back to original directory for next run
    os.chdir(p.current_dir)

    plt.close("all")

    return p.work_subfolder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test_mode",
        help="Run in testing mode, number will g",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-c", "--config_file", help="Provide model parameter file", required=True
    )

    parser.add_argument("-r", "--run_id", help="Run ID", default=None)

    args = parser.parse_args()

    # Create RPM configuration with randomized parameters
    rpm_config = RPMConfig.create_random()
    # Build model using the selected rpm factors
    build_model(
        args.config_file,
        args.run_id,
        args.test_mode,
        rpm_factors=rpm_config.to_dict(),
    )
