# Description

This library provides the data of the paper "An Integrated Predictive Maintenance and Operations Scheduling Framework for Power Systems under Failure Uncertainty" published in INFORMS Journal on Computing, in press.


We conduct our computational study based on the IEEE instances from MATPOWER (Zimmerman et al. 2011). All other parameters are chosen as their original values in the MATPOWER instance for each system component, unless otherwise stated. 


For each MATPOWER instance, the following data is provided in the .csv file:

  * 'instance'_scen_50: failures scenarios of size 50
  
  * 'instance'_scen_100: failures scenarios of size 100
  
  * 'instance'_scen_200: failures scenarios of size 200
  
  * 'instance'_demand: hourly load demand over a week
  
  * 'instance'_cost: generator costs


Maximum voltage angle ${\delta}^{\max}_i$ and minimum voltage angle ${\delta}^{\min}_i$ are chosen as $\pi$ and $-\pi$ for bus $i \in \mathcal{B}$.


Both minimum up time $MU_i$ and minimum down time $MD_i$ are set to $1$ whereas ramp up $RU_i$ and ramp down $RD_i$ rates are chosen as $p_i^{max}$ and $-p_i^{max}$ for generator $i \in \mathcal{G}$.

# References

[R. D. Zimmerman, C. E. Murillo-Sánchez and R. J. Thomas, "MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education," in IEEE Transactions on Power Systems, vol. 26, no. 1, pp. 12-19, Feb. 2011.](https://ieeexplore.ieee.org/document/5491276)

