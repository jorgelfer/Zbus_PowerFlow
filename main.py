import py_dss_interface
from Methods.Ybus import Ybus
from Methods.noLoadV import noLoadV
from Methods.setLoads import setLoads
from Methods.Zbus import Zbus 
import os
import pathlib
 
dssCase = "123Bus"  # name of the IEEE case
dssFile = "IEEE123Master.dss"

scriptPath = os.path.dirname(os.path.abspath(__file__))
dssPath = pathlib.Path(scriptPath).joinpath("IEEETestCases", dssCase, dssFile)

dss = py_dss_interface.DSSDLL()
dss.text(f"Compile [{dssPath}]")

# get system Ybus
Ybus = Ybus(dss, regTypes="non-ideal")
network = Ybus.define_Ybus()
# compute the noLoad volts
noLoadQant = noLoadV(network)
# get load model
load = setLoads(dss, network, noLoadQant)
loadQant = load.get_Load()
# perform the z bus PF
Zbus = Zbus(network, noLoadQant, loadQant)
output_v = Zbus.perform_Zbus()
