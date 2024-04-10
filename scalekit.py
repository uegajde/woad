import numpy as np

# constants copied from "INFO  [CONST_setup] List of constants"
PI = 3.1415926535897931
RADIUS = 6371220.0000000000
OHM = 7.2920000000000000E-005
GRAV = 9.8066499999999994
STB = 5.6703730000000003E-008
KARMAN = 0.40000000000000002
R = 8.3143600000000006
Mdry = 28.966000000000001
Rdry = 287.04000000000002
CPdry = 1004.6400000000000
Cvdry = 717.59999999999991
LAPS = 6.4999999999999997E-003
LAPSdry = 9.7613573021181708E-003
Mvap = 18.015999999999998
Rvap = 461.50000000000000
CPvap = 1846.0000000000000
CVvap = 1384.5000000000000
CL = 4218.0000000000000
CI = 2106.0000000000000
EPSvap = 0.62197183098591557
EPSTvap = 0.60778985507246364
LHV0 = 2501000.0000000000
LHS0 = 2834000.0000000000
LHF0 = 333000.00000000000
LHV00 = 3148911.7999999998
LHS00 = 2905019.0000000000
LHF00 = -243892.79999999993
PSAT0 = 610.77999999999997
DWATR = 1000.0000000000000
DICE = 916.79999999999995
SOUND = 331.31098140568781
Pstd = 101325.00000000000
PRE00 = 100000.00000000000
Tstd = 288.14999999999998
TEM00 = 273.14999999999998

# function copied from ATMOS_HYDROMETEOR_setup and ATMOS_HYDROMETEOR_regist
# in /scale/scalelib/src/atmosphere/common/scale_atmos_hydrometeor.F90
def HYDROMETEOR_regist_for_Tomita08():
    # only for THERMODYN_TYPE='EXACT' and Tomita(2008) scheme
    CV_VAPOR = CVvap
    CP_VAPOR = CPvap
    CV_WATER = CL
    CP_WATER = CV_WATER
    CV_ICE = CI
    CP_ICE = CV_ICE
    LHV = LHV00
    LHF = LHF00

    # based on scale_atmos_phy_mp_tomita08
    # Q variables in Tomita(2008) are 'QV','QC','QR','QI','QS','QG'
    #                              [vapor] [ liquid] [    ice     ]
    CV = np.array([CV_VAPOR, CV_WATER, CV_WATER, CV_ICE, CV_ICE, CV_ICE])  # TRACER_R
    CP = np.array([CP_VAPOR, CP_WATER, CP_WATER, CP_ICE, CP_ICE, CP_ICE])  # TRACER_CV
    R = np.array([Rvap, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # TRACER_CP
    EI0 = np.array([LHV, 0.0, 0.0, - LHF, - LHF, - LHF])  # TRACER_EI0
    TRACER_MASS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return TRACER_MASS, R, CV, CP, EI0

def get_diagVars_from_progVars(ncfile, yxzRange=None):
    # read prognostic variables
    DENS = np.array(ncfile['DENS'])
    MOMZ = np.array(ncfile['MOMZ'])
    MOMY = np.array(ncfile['MOMY'])
    MOMX = np.array(ncfile['MOMX'])
    RHOT = np.array(ncfile['RHOT'])
    QVarNames = ['QV', 'QC', 'QR', 'QI', 'QS', 'QG']
    QVars = []
    for iQVarName in QVarNames:
        QVars.append(np.array(ncfile[iQVarName]))

    # 
    if yxzRange is not None:
        yRange = yxzRange[0]
        xRange = yxzRange[1]
        zRange = yxzRange[2]

        DENS = DENS[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]]
        MOMZ = MOMZ[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]+1]
        MOMY = MOMY[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]]
        MOMX = MOMX[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]]
        RHOT = RHOT[yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]]
        for iqw in range(6):
            QVars[iqw] = QVars[iqw][yRange[0]:yRange[1], xRange[0]:xRange[1], zRange[0]:zRange[1]]

    # diagnose: QDry, Rtot, CVtot, CPtot
    QDry, Rtot, CVtot, CPtot = _ATMOS_THERMODYN_specific_heat_for_Tomita08(QVars)

    # diagnose: PRES, TEMP, POTT, EXNER
    PRES, TEMP, POTT, EXNER = _ATMOS_DIAGNOSTIC_get_therm_rhot(DENS, RHOT, Rtot, CVtot, CPtot)

    # diagnose: PHYD, PHYDH
    # based on ATMOS_DIAGNOSTIC_get_phyd
    # note: skipped

    # diagnose: U, V, W (W is not done yet)
    U, V, W = _ATMOS_DIAGNOSTIC_CARTESC_get_vel(DENS, MOMX, MOMY, MOMZ)

    #
    diagVars = {'QDry': QDry, 'Rtot': Rtot, 'CVtot': CVtot, 'CPtot': CPtot,
                'PRES': PRES, 'TEMP': TEMP, 'POTT': POTT, 'EXNER': EXNER,
                'U': U, 'V': V, 'W':W}
    return diagVars

def _ATMOS_THERMODYN_specific_heat_for_Tomita08(QVars):
    # diagnose: QDry, Rtot, CVtot, CPtot
    # based on ATMOS_THERMODYN_specific_heat
    # note: only for Tomita(2008) scheme
    QDry = 1.0
    Rtot = 0.0
    CVtot = 0.0
    CPtot = 0.0
    Mq, Rq, CVq, CPq, _ = HYDROMETEOR_regist_for_Tomita08()

    for iqw in range(6):
        QDry = QDry - QVars[iqw] * Mq[iqw]  # QTRC_av -> q, TRACER_MASS -> Mq
        Rtot = Rtot + QVars[iqw] * Rq[iqw]  # QTRC_av -> q, TRACER_R -> Rq
        CVtot = CVtot + QVars[iqw] * CVq[iqw]  # QTRC_av -> q, TRACER_CV -> CVq
        CPtot = CPtot + QVars[iqw] * CPq[iqw]  # QTRC_av -> q, TRACER_CP -> CPq

    Rtot = Rtot + QDry * Rdry  # Rdry <-> CONST_RDRY
    CVtot = CVtot + QDry * Cvdry
    CPtot = CPtot + QDry * CPdry  # CPdry <-> CONST_CPDRY
    return QDry, Rtot, CVtot, CPtot

def _ATMOS_DIAGNOSTIC_get_therm_rhot(DENS, RHOT, Rtot, CVtot, CPtot):
    # diagnose: PRES, TEMP, POTT, EXNER
    # based on ATMOS_DIAGNOSTIC_get_therm_rhot
    PRES = PRE00 * (RHOT * Rtot / PRE00)**(CPtot/CVtot)
    TEMP = PRES / (DENS * Rtot)
    POTT = RHOT / DENS
    EXNER = TEMP / POTT
    return PRES, TEMP, POTT, EXNER

def _ATMOS_DIAGNOSTIC_CARTESC_get_vel(DENS, MOMX, MOMY, MOMZ):
    # diagnose: U, V, W
    # based on ATMOS_DIAGNOSTIC_CARTESC_get_vel

    # initial
    yLen, xLen, zLen = DENS.shape
    U = np.zeros((yLen, xLen, zLen))
    V = np.zeros((yLen, xLen, zLen))
    W = np.zeros((yLen, xLen, zLen))

    # U
    U[:, 0, :] = MOMX[:, 0, :] / DENS[:, 0, :]
    U[:, 1:xLen, :] = 0.5 * (MOMX[:, 0:xLen-1, :]+MOMX[:, 1:xLen, :]) / DENS[:, 1:xLen, :]

    # V
    V[0, :, :] = MOMY[0, :, :] / DENS[0, :, :]
    V[1:xLen, :, :] = 0.5 * (MOMY[0:xLen-1, :, :]+MOMY[1:xLen, :, :]) / DENS[1:xLen, :, :]

    # W
    # confusing: MOMZ in restart-output is [y,x,zh] instead [y,x,z] (zh=z+1)
    # Top layer
    # W(KE,i,j) = 0.5_RP * ( MOMZ(KE-1,i,j) ) / DENS(KE,i,j)
    # 
    # Middle layers
    # W(k,i,j) = 0.5_RP * ( MOMZ(k-1,i,j)+MOMZ(k,i,j) ) / DENS(k,i,j)
    # 
    # Lowest layer
    # need to investigate how J13G, J23G, GSQRT be calculated
    #   ! at KS+1/2
    #   momws = MOMZ(KS,i,j) &
    #         + ( J13G(KS,i,j,I_XYW) * ( MOMX(KS,i,j) + MOMX(KS,i-1,j) + MOMX(KS+1,i,j) + MOMX(KS+1,i-1,j) ) &
    #         + J23G(KS,i,j,I_XYW) * ( MOMY(KS,i,j) + MOMY(KS,i,j-1) + MOMY(KS+1,i,j) + MOMY(KS+1,i,j-1) ) ) &
    #         * 0.25_RP / GSQRT(KS,i,j,I_XYW)
    #   ! at KS
    #   ! momws at the surface is assumed to be zero
    #   W(KS,i,j) = ( momws * 0.5_RP                                               &
    #                - ( J13G(KS,i,j,I_XYZ) * ( MOMX(KS,i,j) + MOMX(KS,i-1,j) )    &
    #                  + J23G(KS,i,j,I_XYZ) * ( MOMY(KS,i,j) + MOMY(KS,i,j-1) ) )  &
    #                  * 0.5_RP / GSQRT(KS,i,j,I_XYZ)                              &
    #               ) / DENS(KS,i,j)
    W = 0.5 * ((MOMZ[:, :, 0:zLen]+MOMZ[:, :, 1:zLen+1]) / DENS)

    return U, V, W
