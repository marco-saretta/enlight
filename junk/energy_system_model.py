# Load modules
import gurobipy as gb
import pandas as pd
import numpy as np
import sys
from enlight.data_ops import load_data

class expando(object):
    pass

class model():
    def __init__(self, week, inputpath):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self._load_data(week, inputpath)
        self._aux_data()
        
    def _load_data(self, week, inputpath):
        # Dataframes
        (self.data.onshorewind_df,                  # Hourly onshore wind production
         self.data.offshorewind_df,                 # Hourly offshore wind production
         self.data.solarpv_df,                      # Hourly solar PV production
         self.data.runofriver_df,                   # Hourly run-of-ricer production
         self.data.reshydro_generators_df,          # List of reservoir hydro generators
         self.data.reshydro_energy_df,              # Weekly energy availability  of reservoir hydro (MWh)
         self.data.pumphydro_generators_df,         # List of pumped hydro generators
         self.data.generators_df,                   # List of thermal generators
         self.data.lines_df,                        # List of electrical connectors
         self.data.classic_demand_df,               # Hourly classical demand
         self.data.EV_baseload_df,                  # Hourly EV baseload demand
         self.data.EV_flexload_df,                  # Weekly EV demand to be supplied
         self.data.EV_flexload_maxcapacity_df,      # Maximum capacity of flexible power delivered
         self.data.PtX_demand_df,                   # List of PtX plant
         self.data.heating_demand_df) = load_data(week, inputpath)  # List of electric heating demands
        
        # Sets
        self.data.times = list(self.data.classic_demand_df.index)
        self.data.el_zones = list(self.data.classic_demand_df.columns)
        self.data.generators = list(self.data.generators_df.index)
        self.data.lines = [tuple(x) for x in self.data.lines_df[['fromZone', 'toZone']].to_numpy()]
        self.data.reshydro_generators = list(self.data.reshydro_generators_df.index)
        self.data.pumphydro_generators = list(self.data.pumphydro_generators_df.index)      
        self.data.PtX_units = list(self.data.PtX_demand_df.index)
        self.data.heating_units = list(self.data.heating_demand_df.index)    
        
        # Sets length
        self.data.T = len(self.data.times)
        self.data.Z = len(self.data.el_zones)
        self.data.G = len(self.data.generators) 
        self.data.L = len(self.data.lines)
        self.data.Gres = len(self.data.reshydro_generators)
        self.data.Gpump = len(self.data.pumphydro_generators)       # Store length pumped hydro generators
        self.data.DPtX = len(self.data.PtX_units)
        self.data.Dheat = len(self.data.heating_units)
        
        # Onshore wind bid price
        self.data.Windon_bidprice = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                          columns=self.data.times, 
                                                          data = 0.01))
        
        # Onshore wind bid volume
        self.data.Windon_bidvol = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                        columns=self.data.times, 
                                                        data = self.data.onshorewind_df.values.T))
        
        # Offshore wind bid price
        self.data.Windof_bidprice = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                          columns=self.data.times, 
                                                          data = 0))
        
        # Offshore wind bid volume
        self.data.Windof_bidvol = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                        columns=self.data.times, 
                                                        data = self.data.offshorewind_df.values.T))
                
        # Solar PV bid price
        self.data.SolarPV_bidprice = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                          columns=self.data.times, 
                                                          data = -0.03))
        
        # Solar PV bid volume
        self.data.SolarPV_bidvol = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                         columns=self.data.times, 
                                                         data = self.data.solarpv_df.values.T))
                
        # Run-of-river bid price
        self.data.Runofriver_bidprice = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                          columns=self.data.times, 
                                                          data = -0.02))
        
        # Run-of-river bid volume
        self.data.Runofriver_bidvol = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                            columns=self.data.times, 
                                                            data = self.data.runofriver_df.values.T))
        
        # Reservoir hydro generator bid price
        self.data.Reshydro_bidprice = np.array(pd.DataFrame(index=self.data.reshydro_generators, 
                                                            columns=self.data.times, 
                                                            data = self.data.reshydro_generators_df.prodcost.values.reshape(self.data.Gres,1)*np.ones((self.data.Gres,self.data.T))))
        
        # Reservoir hydro generator bid volume
        self.data.Reshydro_bidvol = np.array(pd.DataFrame(index=self.data.reshydro_generators, 
                                                          columns=self.data.times, 
                                                          data = self.data.reshydro_generators_df.capacity_el.values.reshape(self.data.Gres,1)*np.ones((self.data.Gres,self.data.T))))
        
        # Reservoir hydro generator available production
        Available_prod = np.array(self.data.reshydro_energy_df)
        self.data.Reshydro_availableprod = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                                 columns=['Available_prod'], 
                                                                 data = Available_prod)) 
        
        # Pumped hydro generator bid price - CHARGING MODE
        self.data.Pumphydro_cha_bidprice = np.array(pd.DataFrame(index=self.data.pumphydro_generators, 
                                                            columns=self.data.times, 
                                                            data = self.data.pumphydro_generators_df.Pumped_cons.values.reshape(self.data.Gpump,1)*np.ones((self.data.Gpump,self.data.T))))
        
        # Pumped hydro generator bid volume
        self.data.Pumphydro_cha_bidvol = np.array(pd.DataFrame(index=self.data.pumphydro_generators, 
                                                          columns=self.data.times, 
                                                          data = self.data.pumphydro_generators_df.pumping_capacity.values.reshape(self.data.Gpump,1)*np.ones((self.data.Gpump,self.data.T))))
 
        # Pumped hydro generator bid price - DISCHARGING MODE
        self.data.Pumphydro_dis_bidprice = np.array(pd.DataFrame(index=self.data.pumphydro_generators, 
                                                            columns=self.data.times, 
                                                            data = self.data.pumphydro_generators_df.Pumped_prod.values.reshape(self.data.Gpump,1)*np.ones((self.data.Gpump,self.data.T))))
                                                            
        # Pumped hydro generator bid volume
        self.data.Pumphydro_dis_bidvol = np.array(pd.DataFrame(index=self.data.pumphydro_generators, 
                                                          columns=self.data.times, 
                                                          data = self.data.pumphydro_generators_df.capacity_el.values.reshape(self.data.Gpump,1)*np.ones((self.data.Gpump,self.data.T))))
                                                          
        # Pumped hydro generator SOC max
        self.data.Pumphydro_SOC_max = np.array(pd.DataFrame(index=self.data.pumphydro_generators, 
                                                                 columns=self.data.times, 
                                                                 data = self.data.pumphydro_generators_df.Storage_Capacity.values.reshape(self.data.Gpump,1)*np.ones((self.data.Gpump,self.data.T))))
        
        # Generator bid price
        self.data.Generators_bidprice = np.array(pd.DataFrame(index=self.data.generators,
                                                              columns=self.data.times, 
                                                              data = self.data.generators_df.prodcost.values.reshape(self.data.G,1)*np.ones((self.data.G,self.data.T))))
        
        # Generator bid volume
        self.data.Generators_bidvol = np.array(pd.DataFrame(index=self.data.generators,
                                                            columns=self.data.times, 
                                                            data = self.data.generators_df.capacity_el.values.reshape(self.data.G,1)*np.ones((self.data.G,self.data.T))))
           
        # Classic demand VOLL
        self.data.Classicdem_bidprice = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                              columns=self.data.times, 
                                                              data = 5000))
        
        # Classic demand bid volume
        self.data.Classicdem_bidvol = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                            columns=self.data.times, 
                                                            data = self.data.classic_demand_df.values.T))
        
        # EV baseload demand VOLL
        self.data.EV_baseload_bidprice = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                              columns=self.data.times, 
                                                              data = 5000))
        
        # EV baseload bid volume
        self.data.EV_baseload_bidvol = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                            columns=self.data.times, 
                                                            data = self.data.EV_baseload_df.values.T))
        

        # EV flexload demand VOLL
        self.data.EV_flexload_bidprice = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                              columns=self.data.times, 
                                                              data = 100))         

        # EV flexload bid volume
        EV_flexload = np.array(self.data.EV_flexload_df)
        self.data.EV_flexload_mindem = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                            columns=['EV_peak_demand'], 
                                                            data = EV_flexload))
        
        # EV flexload max capacity volume
        EV_flexload_maxcapacity = np.array(self.data.EV_flexload_maxcapacity_df)
        self.data.EV_flexload_maxcapacity_ub = np.array(pd.DataFrame(index=self.data.el_zones, 
                                                            columns=['EV_flexload_maxcapacity'], 
                                                            data = EV_flexload_maxcapacity))

       # PtX demand bid price
        self.data.PtX_bidprice = np.array(pd.DataFrame(index=self.data.PtX_units,
                                                              columns=self.data.times, 
                                                              data = self.data.PtX_demand_df['Demand price'].values.reshape(self.data.DPtX,1)*np.ones((self.data.DPtX,self.data.T))))
        
        # PtX demand bid volume
        self.data.PtX_bidvol = np.array(pd.DataFrame(index=self.data.PtX_units,
                                                            columns=self.data.times, 
                                                            data = self.data.PtX_demand_df['Electric capacity'].values.reshape(self.data.DPtX,1)*np.ones((self.data.DPtX,self.data.T))))

       # El. Heating units demand bid price
        self.data.ElHeat_bidprice = np.array(pd.DataFrame(index=self.data.heating_units,
                                                              columns=self.data.times, 
                                                              data = self.data.heating_demand_df.demand_price.values.reshape(self.data.Dheat,1)*np.ones((self.data.Dheat,self.data.T))))
        
        # El. Heating units demand bid volume
        self.data.ElHeat_bidvol = np.array(pd.DataFrame(index=self.data.heating_units,
                                                            columns=self.data.times, 
                                                            data = self.data.heating_demand_df.Electric_capacity.values.reshape(self.data.Dheat,1)*np.ones((self.data.Dheat,self.data.T))))

        # Line limits
        self.data.LinescapAB = (np.ones((self.data.T,self.data.L)) * np.array(self.data.lines_df['Capacity_AB'])).T
        self.data.LinescapBA = (np.ones((self.data.T,self.data.L)) * np.array(self.data.lines_df['Capacity_BA'])).T
        
    def _aux_data(self):
        Z = self.data.Z
        G = self.data.G
        Gres = self.data.Gres
        Gpump = self.data.Gpump         # Add list of generators
        DPtX = self.data.DPtX
        Dheat = self.data.Dheat
        L = self.data.L
        
        # Assigning generators to zones
        self.data.G_Z = np.zeros((G,Z))
        for i in range(Z):
            self.data.G_Z[:,i] = np.array(self.data.generators_df.zone_el==self.data.el_zones[i]).astype(int)
        
        # Assigning reservoir hydro generators to zones
        self.data.Gres_Z = np.zeros((Gres,Z))
        for i in range(Z):
            self.data.Gres_Z[:,i] = np.array(self.data.reshydro_generators_df.zone_el==self.data.el_zones[i]).astype(int)
        
        # Assigning pumped hydro generators to zones
        self.data.Gpump_Z = np.zeros((Gpump,Z))
        for i in range(Z):
            self.data.Gpump_Z[:,i] = np.array(self.data.pumphydro_generators_df.zone_el==self.data.el_zones[i]).astype(int)
        
        # Assigning heating units to zones
        self.data.DPtX_Z = np.zeros((DPtX,Z))
        for i in range(Z):
            self.data.DPtX_Z[:,i] = np.array(self.data.PtX_demand_df.zone_el==self.data.el_zones[i]).astype(int)
       
       # Assigning heating units to zones
        self.data.Dheat_Z = np.zeros((Dheat,Z))
        for i in range(Z):
            self.data.Dheat_Z[:,i] = np.array(self.data.heating_demand_df.zone_el==self.data.el_zones[i]).astype(int)
        
        
        # Assigning linelimits
        self.data.L_Z = np.zeros((L,Z))
        for i in range(L):
            for j in range(Z):
                if self.data.el_zones[j] == self.data.lines[i][0]:
                    self.data.L_Z[i,j] = 1
                elif self.data.el_zones[j] == self.data.lines[i][1]:
                    self.data.L_Z[i,j] = -1
        
    def _build_model(self):
        self.m = gb.Model()
        self._build_variables()   
        self._build_objective_function()
        self._build_constraints()
        
    def _build_variables(self):
        T = self.data.T
        Z = self.data.Z
        G = self.data.G
        Gres = self.data.Gres
        Gpump = self.data.Gpump         # Add list of pumped hydro plants
        Dheat = self.data.Dheat
        DPtX = self.data.DPtX
        L = self.data.L
        
        Windon_bidvol = self.data.Windon_bidvol
        Windof_bidvol = self.data.Windof_bidvol
        SolarPV_bidvol = self.data.SolarPV_bidvol
        Runofriver_bidvol = self.data.Runofriver_bidvol
        Reshydro_bidvol = self.data.Reshydro_bidvol
        Pumphydro_dis_bidvol = self.data.Pumphydro_dis_bidvol       # Add variable discharge pumped hydro power plant
        Pumphydro_cha_bidvol = self.data.Pumphydro_cha_bidvol       # Add variable charge pumped hydro power plant
        Pumphydro_SOC_max = self.data.Pumphydro_SOC_max             # Add variable for reservoir level
        Generators_bidvol = self.data.Generators_bidvol
        Classicdem_bidvol = self.data.Classicdem_bidvol
        EV_baseload_bidvol = self.data.EV_baseload_bidvol
        EV_flexload_maxcapacity_ub = self.data.EV_flexload_maxcapacity_ub
        PtX_bidvol = self.data.PtX_bidvol
        ElHeat_bidvol = self.data.ElHeat_bidvol
        
        
        LinescapAB = self.data.LinescapAB
        LinescapBA = self.data.LinescapBA
        
        # Onshore wind production
        self.variables.windon_prod = self.m.addMVar((Z,T), lb = np.zeros((Z,T)), 
                                                    ub = Windon_bidvol,
                                                    name='windon_prod')
        
        # Offshore wind production
        self.variables.windof_prod = self.m.addMVar((Z,T), lb = np.zeros((Z,T)), 
                                                   ub = Windof_bidvol,
                                                   name='windof_prod')
        
        # Solar PV production
        self.variables.solarpv_prod = self.m.addMVar((Z,T), lb = np.zeros((Z,T)), 
                                                   ub = SolarPV_bidvol,
                                                   name='solarpv_prod')
        
        # Run-of-river production
        self.variables.runofriver_prod = self.m.addMVar((Z,T), lb = np.zeros((Z,T)), 
                                                   ub = Runofriver_bidvol,
                                                   name='runofriver_prod')
        
        # Capacity on reservoir hydro generators
        self.variables.reshydro_prod = self.m.addMVar((Gres,T), lb = np.zeros((Gres,T)), 
                                                   ub = Reshydro_bidvol,
                                                   name='reshydro_prod')          
      
        # Capacity on pumped hydro generators
        self.variables.pumphydro_dis_prod = self.m.addMVar((Gpump,T), lb = np.zeros((Gpump,T)), 
                                                   ub = Pumphydro_dis_bidvol,
                                                   name='pumphydro_dis_bidvol')  
        
        # Capacity on pumped hydro DEMAND
        self.variables.pumphydro_cha_demand = self.m.addMVar((Gpump,T), lb = np.zeros((Gpump,T)), 
                                                   ub = Pumphydro_cha_bidvol,
                                                   name='pumphydro_cha_bidvol') 
        
        # Pumped hydro SOC
        self.variables.pumphydro_SOC = self.m.addMVar((Gpump,T), lb = np.zeros((Gpump,T)), 
                                                   ub = Pumphydro_SOC_max,                   
                                                   name='pumphydro_level')  
       
        # Capacity on generators
        self.variables.generators_prod = self.m.addMVar((G,T), lb = np.zeros((G,T)), 
                                                        ub = Generators_bidvol,
                                                        name='generators_prod')
                
        # Classic electricity demand
        self.variables.classic_dem = self.m.addMVar((Z,T), lb = np.zeros((Z,T)), 
                                                   ub = Classicdem_bidvol,
                                                   name='classic_dem')
        
        # EV baseload demand
        self.variables.EV_baseload_dem = self.m.addMVar((Z,T), lb = np.zeros((Z, T)),
                                                   ub = EV_baseload_bidvol,
                                                   name = 'EV_baseload_demand')        
        
        # EV flexload demand
        self.variables.EV_flexible_dem = self.m.addMVar((Z,T), lb = np.zeros((Z, T)),
                                                   ub = np.ones((Z, T)) * EV_flexload_maxcapacity_ub,
                                                   name = 'EV_flexible_dem')        
        
        # Electric heating demand
        self.variables.ElHeat_dem = self.m.addMVar((Dheat, T), lb = np.zeros((Dheat, T)),
                                                   ub = ElHeat_bidvol,
                                                   name = 'elheat_demand')
        # PtX demand
        self.variables.PtX_dem = self.m.addMVar((DPtX, T), lb = np.zeros((DPtX, T)),
                                                   ub = PtX_bidvol,
                                                   name = 'PtX_dem')
    
        # Electricity export
        self.variables.export = self.m.addMVar((Z,T), lb = -gb.GRB.INFINITY * np.ones((Z,T)), 
                                               ub = gb.GRB.INFINITY * np.ones((Z,T)),
                                               name = 'export')
        
        # Electricity line limits      
        self.variables.lineflow = self.m.addMVar((L,T), lb = -LinescapBA,
                                                 ub = LinescapAB, name='lineflow')
        
    def _build_constraints(self):
        T = self.data.T
        Z = self.data.Z
        G_Z = self.data.G_Z
        Gres_Z = self.data.Gres_Z
        Gpump_Z = self.data.Gpump_Z
        Dheat_Z = self.data.Dheat_Z
        DPtX_Z = self.data.DPtX_Z
        L_Z = self.data.L_Z
        
        Reshydro_availableprod = self.data.Reshydro_availableprod
        Reshydro_availableprod = self.data.Reshydro_availableprod
        Pumphydro_SOC_max = self.data.Pumphydro_SOC_max
        EV_flexload_mindem = self.data.EV_flexload_mindem
        
        windon_prod = self.variables.windon_prod
        windof_prod = self.variables.windof_prod
        solarpv_prod = self.variables.solarpv_prod
        runofriver_prod = self.variables.runofriver_prod
        reshydro_prod = self.variables.reshydro_prod
        pumphydro_dis_prod = self.variables.pumphydro_dis_prod                  
        pumphydro_cha_demand = self.variables.pumphydro_cha_demand              
        pumphydro_SOC = self.variables.pumphydro_SOC                            
        generators_prod = self.variables.generators_prod
        classic_dem = self.variables.classic_dem
        EV_baseload_dem = self.variables.EV_baseload_dem
        EV_flexible_dem = self.variables.EV_flexible_dem
        ElHeat_dem = self.variables.ElHeat_dem
        PtX_dem = self.variables.PtX_dem
        export = self.variables.export
        lineflow = self.variables.lineflow
        
        # Power balance constraint
        self.constraints.powerbalance = self.m.addConstrs(
            (G_Z.T @ generators_prod[:,t] + windon_prod[:,t]  + windof_prod[:,t]  + solarpv_prod[:,t]  + runofriver_prod[:,t] + Gres_Z.T @ reshydro_prod[:,t] + Gpump_Z.T @ pumphydro_dis_prod[:,t]
             == Gpump_Z.T @ pumphydro_cha_demand[:,t] + classic_dem[:,t] +EV_flexible_dem[:,t] +EV_baseload_dem[:,t] + DPtX_Z.T @ PtX_dem[:,t] + Dheat_Z.T @ ElHeat_dem[:,t] +export[:,t] for t in range(T)), name='powerbalance')
        
        # Export constraint
        self.constraints.exports = self.m.addConstrs(
            (export[:,t]
            ==
            L_Z.T @ lineflow[:,t] 
            for t in range(T)), name='exports')
        
        # Reservoir hydro constraint
        self.constraints.maxhydrores = self.m.addConstr(
            sum(Gres_Z.T @ reshydro_prod[:,t] for t in range(T))
            <=
            Reshydro_availableprod.reshape(Z), name='maxhydrores')
        
        self.constraints.maxflexev = self.m.addConstrs(
            (sum(EV_flexible_dem[z,t] for t in range(T)) <= EV_flexload_mindem[z] for z in range(Z)),  name='maxflexev')


        eff_cha = 0.9
        eff_dis = 0.9
        # Pumped hydro initial SOC constraint 
        self.constraints.pumphydro_SOC_start = self.m.addConstr(pumphydro_SOC[:,0] == Pumphydro_SOC_max[:,0]/2 + pumphydro_cha_demand[:, 0] * eff_cha
                                                                                       - pumphydro_dis_prod[:, 0] / eff_dis, name='pumphydro_SOC_start')
        
        self.constraints.pumphydro_SOC_general = self.m.addConstr(pumphydro_SOC[:,1:] ==  pumphydro_cha_demand[:, 1:] * eff_cha
                                                                  - pumphydro_dis_prod[:, 1:]/ eff_dis + pumphydro_SOC[:, :-1], 
                                                                  name='pumphydro_SOC_general')
        
        self.constraints.pumphydro_SOC_start = self.m.addConstr(pumphydro_SOC[:,-1] == Pumphydro_SOC_max[:,-1]/2 , name='pumphydro_SOC_start')

        
    def _build_objective_function(self):
        T = self.data.T

        Windon_bidprice = self.data.Windon_bidprice
        Windof_bidprice = self.data.Windof_bidprice
        SolarPV_bidprice = self.data.SolarPV_bidprice
        Runofriver_bidprice = self.data.Runofriver_bidprice       
        Reshydro_bidprice = self.data.Reshydro_bidprice      
        Generators_bidprice = self.data.Generators_bidprice
        Classicdem_bidprice = self.data.Classicdem_bidprice
        Pumphydro_dis_bidprice = self.data.Pumphydro_dis_bidprice
        Pumphydro_cha_bidprice = self.data.Pumphydro_cha_bidprice
        EV_baseload_bidprice = self.data.EV_baseload_bidprice
        EV_flexload_bidprice = self.data.EV_flexload_bidprice
        ElHeat_bidprice = self.data.ElHeat_bidprice
        PtX_bidprice = self.data.PtX_bidprice
        
        windon_prod = self.variables.windon_prod
        windof_prod = self.variables.windof_prod
        solarpv_prod = self.variables.solarpv_prod
        runofriver_prod = self.variables.runofriver_prod
        reshydro_prod = self.variables.reshydro_prod
        pumphydro_dis_prod = self.variables.pumphydro_dis_prod              # Add pumped hydro variables
        pumphydro_cha_demand = self.variables.pumphydro_cha_demand          # Add pumped hydro variables
        EV_baseload_dem = self.variables.EV_baseload_dem
        EV_flexible_dem = self.variables.EV_flexible_dem
        ElHeat_dem   = self.variables.ElHeat_dem
        generators_prod = self.variables.generators_prod
        classic_dem = self.variables.classic_dem
        PtX_dem = self.variables.PtX_dem
        
        # Objective function
        self.expr_obj = sum(
            - Classicdem_bidprice[:,t] @ classic_dem[:,t]
            - EV_baseload_bidprice[:,t] @ EV_baseload_dem[:,t]
            - EV_flexload_bidprice[:,t] @ EV_flexible_dem[:,t]
            - PtX_bidprice[:,t] @ PtX_dem[:,t]
            - ElHeat_bidprice[:,t] @ ElHeat_dem[:,t]
            - Pumphydro_cha_bidprice[:,t] @ pumphydro_cha_demand[:,t]
            + Windon_bidprice[:,t] @ windon_prod[:,t]
            + Windof_bidprice[:,t] @ windof_prod[:,t]
            + SolarPV_bidprice[:,t] @ solarpv_prod[:,t]
            + Runofriver_bidprice[:,t] @ runofriver_prod[:,t]
            + Generators_bidprice[:,t] @ generators_prod[:,t]
            + Reshydro_bidprice[:,t] @ reshydro_prod[:,t]
            + Pumphydro_dis_bidprice[:,t] @ pumphydro_dis_prod[:,t]
            for t in range(T))
        self.m.setObjective(self.expr_obj, gb.GRB.MINIMIZE)
        
    def _run_model(self):
        self._build_model()
        self.m.optimize()
        self.m.update()
        if self.m.solCount == 0:
            self.m.computeIIS()
            self.m.write("model_iis.ilp")
            sys.exit('Code stopped. Infeasible model')
        self.m.write("model.lp")