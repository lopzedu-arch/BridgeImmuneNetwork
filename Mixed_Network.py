import numpy as np
from scipy.integrate import solve_ivp

class NetworkEquations:
    def __init__(self):
        self.array_of_elements = None
        self.n = None
        self.out_of_network_arguments = None
        self.out_of_network_names = None
        self.Nodes()
        self.networkparameters()
        self.create_ode_system()
        self.solve()

    def Nodes(self):
        self.array_of_elements = np.array(['IFNGR', 'CSF2RA', 'IL1R', 'TLR4', 'FCGR', 'IL4RA', 'IL10R', 'STAT1', 'MSTAT5', 'MNFKB',
                                           'PPARG', 'STAT6', 'JMJD3', 'STAT3', 'SOCS3', 'IRF3', 'ERK', 'KLF4', 'SOCS1', 'IRF4',
                                           'IL1BOUT', 'IL12OUT', 'TAK123', 'JNKMAPK', 'ERK12', 'MSK12', 'GSK3B', 'IL10OUT', 'AKT1', 'AKT3',
                                           'TLR2', 'TLR3', 'TLR7', 'TLR8', 'TLR9', 'MYD88', 'IRAK4', 'IRAK12', 'TRAF6', 'TAB23TAK',
                                           'MKK36', 'P38', 'CREB1', 'MKK47', 'JNK', 'CJUN', 'MKK12', 'CFOS', 'MAP1', 'NEMOIKKB',
                                           'RIP1', 'TLR4END', 'TRIF', 'TRAF3', 'TBK1IKKI', 'IKKA', 'IRF7', 'IL6OUT', 'IL18OUT', 'IL33OUT',
                                           'IFNA', 'PHAGOCYTOSIS', 'PHAGOSOME', 'PROCESSING', 'MHC2', 'CD8086', 'ITAM', 'SYK', 'CARD9', 'VAV',
                                           'RAC', 'CDC42', 'WASP', 'WAVE', 'MCA', 'LYN', 'FYN', 'DAP12', 'MPI3K', 'MPLC',
                                           'MAKT', 'MPKC', 'CD11BCD18', 'HCK', 'FGR', 'FCAR', 'MMTOR', 'MMTORC1', 'MMTORC2', 'MLKB1',
                                           'MAMPK', 'MGlycolysis', 'MOXPHOS', 'MAMPATPratio', 'MHIF1A', 'RIG1', 'MAVS', 'TRADD', 'NEMOIKKAB','NEMOTBK1IKKE',
                                           'TNFR1', 'TRAF2', 'TNFAOUT', 'TCR', 'CD28', 'AP1', 'CD25', 'IL2G', 'IL2E', 'MTOR',
                                           'ZAP70', 'STAT5', 'NFAT', 'NFKB', 'AKT', 'CTLA4', 'CTLA4DIM', 'BCL2', 'NDRG1', 'DAG',
                                           'SOS', 'RASGTPR', 'LCK', 'PDK1', 'LAT', 'PLC', 'PI3K', 'PIP2', 'PIP3', 'IP3',
                                           'CA', 'PKC', 'TBET', 'IFNG', 'GATA3', 'IL4', 'FOXP3', 'IL10', 'TGFB','RORGT',
                                           'IL21', 'IL17', 'BCL6', 'IL9', 'CD40L', 'MTORC1', 'MTORC2', 'LKB1', 'AMPK','Glycolysis',
                                           'GLUTAMINOLISIS', 'AKG', 'OXPHOS', 'AMPATPratio', 'HIF1A'])
        self.n = len(self.array_of_elements)
        self.initial_conditions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
        return self.array_of_elements

    def networkparameters(self):
        self.out_of_network_arguments = np.array(['b', 'LP', 'DSRNA', 'SSRNA', 'CPGDNA', 'IFNGE', 'IFNGL', 'MHC2E', 'CD8086E', 'GMCSFE',
                                                  'IL1BE', 'LPSE', 'IL4E', 'IL4L', 'TGFBE', 'IL10E', 'IL10L', 'IL12E', 'IC3B', 'IGGC',
                                                  'IGAC', 'IL6E', 'IL21E', 'IL18E', 'IL33E', 'MGLC', 'GLC', 'CITDSRNA', 'CITSSRNA', 'TNFA',
                                                  'CIAP', 'GLN', 'MFA', 'FA', 'TRP', 'MO2', 'O2', 'METF', 'RAPA', 'PRED',
                                                  'CS', 'TCD8086', 'TMHC2'])
        self.out_of_network_names = np.array(['Saturation rate', 'Lipoproteins (LP)', 'Double-stranded RNA (dsRNA)', 'Single-stranded RNA (ssRNA)', 'CpG DNA (unmethylated bacterial DNA)', 'Interferon gamma (external)', 'Interferon gamma (local)', 'MHC class II (external)', 'CD80/CD86 (external)', 'Granulocyte-macrophage colony-stimulating factor (GM-CSF)', 'Interleukin-1 beta (external)', 'LPS (Lipopolysaccharide)', 'Interleukin-4 (external)', 'Interleukin-4 (local)', 'Transforming growth factor beta (TGF-β)', 'Interleukin-10 (external)', 'Interleukin-10 (local)', 'Interleukin-12 (local)', 'Complement component iC3b', 'Immunoglobulin G (IgG immune complexes)', 'Immunoglobulin A (IgA immune complexes)', 'Interleukin-6 (local)', 'Interleukin-21 (local)', 'Interleukin-18 (local)', 'Interleukin-33 (local)', 'Glucose (macrophage)', 'Glucose', 'Cytosolic double-stranded RNA', 'Cytosolic single-stranded RNA', 'Tumor necrosis factor alpha (TNF-α)', 'Cellular inhibitors of apoptosis proteins (cIAP)', 'Glutamine', 'Macrophage fatty acids', 'Fatty acids', 'Tryptophan', 'Molecular oxygen (macrophage)', 'Molecular oxygen', 'Methionine flux', 'Rapamycin (mTOR inhibitor)', 'Prednisone (glucocorticoid)', 'Cholesterol synthesis / cellular sterols', 'CD28-CD80/CD86 interaction time', 'TCR-MHC II interaction time'])
        self.initial_out=np.array([8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0.0, 30, 25])
        return self.out_of_network_arguments

    def create_ode_system(self, af=None, decay_rates=None, threshold_values=None):
        def act_f(a_function=None, x=None, b=None, u=None):
            if a_function is None:
                a_function = 'sigmoid'
            if a_function == 'sigmoid':
                return 1 / (1 + np.exp(-b*(x-u)))
            elif a_function == 'tanh':
                return (1+np.tanh(-b*(x-u)))/2
            elif a_function == 'hill':
                return hill_activation(x, n=10, Kd=0.5, u=0.5)
        def hill_activation(x, n=10, Kd=0.5, u=0.5):
            x = np.asarray(x)
            return x**n / (Kd**n + x**n)
        def system_ode(t, q, *p):
            w = np.zeros(len(q))
            w[0] = (p[5]+q[133]-p[5]*q[133])
            w[1] = p[9]
            w[2] = (p[10]+q[20]-p[10]*q[20])
            w[3] = p[11]
            w[4] = ((p[19]+p[19]*p[10]-p[19]*p[19]*p[10])+q[30]-(p[19]+p[19]*p[10]-p[19]*p[19]*p[10])*q[30])
            w[5] = (p[12]+q[135]-p[12]*q[135])
            w[6] = ((p[15]+q[27]-p[15]*q[27])+q[137]-(p[15]+q[27]-p[15]*q[27])*q[137])
            w[7] = q[0]*(1-(q[18]+q[13]-q[18]*q[13]))
            w[8] = q[1]*(1-((q[13]+q[19]-q[13]*q[19])+q[18]-(q[13]+q[19]-q[13]*q[19])*q[18]))
            w[9] = (((((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])+q[81]-(((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])*q[81])+q[50]*q[101]*p[30]-((((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])+q[81]-(((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])*q[81])*q[50]*q[101]*p[30])*(1-(((q[13]+q[10]-q[13]*q[10])+q[17]-(q[13]+q[10]-q[13]*q[10])*q[17])+q[42]-((q[13]+q[10]-q[13]*q[10])+q[17]-(q[13]+q[10]-q[13]*q[10])*q[17])*q[42]))
            w[10] = q[5]
            w[11] = q[5]*(1-q[7])
            w[12] = q[5]
            w[13] = (q[6]+q[16]*q[41]-q[6]*q[16]*q[41])*(1-(q[10]+q[14]-q[10]*q[14]))
            w[14] = (q[9]+q[7]-q[9]*q[7])
            w[15] = (q[54]+q[99]-q[54]*q[99])
            w[16] = ((q[4]+q[46]-q[4]*q[46])+q[78]-(q[4]+q[46]-q[4]*q[46])*q[78])
            w[17] = q[11]
            w[18] = (q[11]+q[60]-q[11]*q[60])
            w[19] = q[12]
            w[20] = (q[9]+q[9]*q[48]-q[9]*q[9]*q[48])
            w[21] = ((q[7]+q[8]*q[9]-q[7]*q[8]*q[9])+q[48]*q[9]-(q[7]+q[8]*q[9]-q[7]*q[8]*q[9])*q[48]*q[9])
            w[22] = q[35]
            w[23] = q[22]
            w[24] = q[23]
            w[25] = (q[24]+q[41]-q[24]*q[41])
            w[26] = ((1-q[29])+(1-q[28])-(1-q[29])*(1-q[28]))
            w[27] = ((((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])+q[13]-((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])*q[13])+q[42]*q[48]-(((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])+q[13]-((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])*q[13])*q[42]*q[48])
            w[28] = q[78]*(1-q[0])
            w[29] = q[78]*(1-q[0])
            w[30] = p[1]
            w[31] = p[2]
            w[32] = p[3]
            w[33] = p[3]
            w[34] = p[4]
            w[35] = ((((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])+q[33]-((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])*q[33])+q[34]-(((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])+q[33]-((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])*q[33])*q[34])
            w[36] = q[35]
            w[37] = q[36]
            w[38] = q[37]
            w[39] = (q[38]+q[50]-q[38]*q[50])
            w[40] = q[39]
            w[41] = q[40]
            w[42] = q[25]*(1-q[26])
            w[43] = q[39]
            w[44] = (q[43]+q[78]-q[43]*q[78])
            w[45] = q[44]
            w[46] = q[49]
            w[47] = q[16]
            w[48] = ((q[45]+q[47]-q[45]*q[47])+q[25]*(1-q[26])-(q[45]+q[47]-q[45]*q[47])*q[25]*(1-q[26]))
            w[49] = q[39]
            w[50] = (q[52]+q[97]-q[52]*q[97])
            w[51] = q[3]
            w[52] = (q[31]+q[51]-q[31]*q[51])
            w[53] = ((((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])+q[37]*q[33]-((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])*q[37]*q[33])+q[37]*q[34]-(((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])+q[37]*q[33]-((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])*q[37]*q[33])*q[37]*q[34])
            w[54] = q[53]*q[52]
            w[55] = q[53]*q[37]
            w[56] = (q[55]+q[99]-q[55]*q[99])
            w[57] = (q[9]+q[48]-q[9]*q[48])
            w[58] = q[9]*q[48]
            w[59] = q[9]*q[48]
            w[60] = (q[15]+q[56]-q[15]*q[56])
            w[61] = (((q[73]*q[72]+q[70]-q[73]*q[72]*q[70])+q[41]*q[30]-(q[73]*q[72]+q[70]-q[73]*q[72]*q[70])*q[41]*q[30])+q[41]*q[34]-((q[73]*q[72]+q[70]-q[73]*q[72]*q[70])+q[41]*q[30]-(q[73]*q[72]+q[70]-q[73]*q[72]*q[70])*q[41]*q[30])*q[41]*q[34])
            w[62] = q[61]
            w[63] = (q[62]+q[51]-q[62]*q[51])
            w[64] = q[63]
            w[65] = q[64]
            w[66] = q[4]
            w[67] = (((q[66]+q[74]-q[66]*q[74])+q[77]-(q[66]+q[74]-q[66]*q[74])*q[77])+q[83]*q[84]-((q[66]+q[74]-q[66]*q[74])+q[77]-(q[66]+q[74]-q[66]*q[74])*q[77])*q[83]*q[84])
            w[68] = q[67]
            w[69] = q[67]
            w[70] = q[69]
            w[71] = q[69]
            w[72] = q[70]
            w[73] = q[71]
            w[74] = q[79]
            w[75] = q[85]
            w[76] = q[85]
            w[77] = (q[76]+q[75]-q[76]*q[75])
            w[78] = (q[67]+q[35]-q[67]*q[35])
            w[79] = q[67]
            w[80] = q[78]
            w[81] = q[79]
            w[82] = p[18]
            w[83] = q[82]
            w[84] = q[82]
            w[85] = p[20]
            w[86] = (q[80]+q[9]-q[80]*q[9])
            w[87] = q[86]*(q[80]+q[9]-q[80]*q[9])*(1-q[90])
            w[88] = (q[86]*q[90]+q[86]*(p[12]+q[135]-p[12]*q[135])-q[86]*q[90]*q[86]*(p[12]+q[135]-p[12]*q[135]))
            w[89] = ((q[80]*q[93]+(p[12]+q[135]-p[12]*q[135])-q[80]*q[93]*(p[12]+q[135]-p[12]*q[135]))+((p[15]+q[137]-p[15]*q[137])+q[27]-(p[15]+q[137]-p[15]*q[137])*q[27])-(q[80]*q[93]+(p[12]+q[135]-p[12]*q[135])-q[80]*q[93]*(p[12]+q[135]-p[12]*q[135]))*((p[15]+q[137]-p[15]*q[137])+q[27]-(p[15]+q[137]-p[15]*q[137])*q[27]))
            w[90] = (q[89]+(q[74]+q[80]-q[74]*q[80])*q[93]-q[89]*(q[74]+q[80]-q[74]*q[80])*q[93])*(1-q[87])
            w[91] = (q[87]+q[94]-q[87]*q[94])*p[25]
            w[92] = q[90]*p[32]
            w[93] = q[91]*(1-q[92])
            w[94] = (1-p[35])*q[80]
            w[95] = (p[27]+p[28]-p[27]*p[28])
            w[96] = q[95]
            w[97] = (q[96]+q[100]-q[96]*q[100])
            w[98] = q[50]*q[97]
            w[99] = q[53]*q[97]
            w[100] = (p[29]+q[102]-p[29]*q[102])
            w[101] = q[97]
            w[102] = q[9]
            w[103] = ((q[64]+p[7]-q[64]*p[7]) / (1 + np.exp((t - p[42]))))*(1-q[116])
            w[104] = ((q[65]+p[8]-q[65]*p[8]) / (1 + np.exp((t - p[41]))))*(1-q[116])
            w[105] = q[121]*(1-p[39])
            w[106] = q[107]*(1-q[116])
            w[107] = q[112]*q[105]*(1-q[118])
            w[108] = q[107]
            w[109] = (q[106]+q[114]-q[106]*q[114])
            w[110] = q[103]*q[122]*(1-q[116])*(1-p[39])
            w[111] = q[106]*(1-q[116])
            w[112] = q[130]*(1-p[40])
            w[113] = q[131]*(1-p[39])
            w[114] = (q[104]*(1-q[116])+q[123]-q[104]*(1-q[116])*q[123])
            w[115] = q[107]*q[110]
            w[116] = (q[115]+q[136]*q[138]-q[115]*q[136]*q[138])
            w[117] = q[114]
            w[118] = q[112]*(1-q[114])
            w[119] = q[125]*q[127]
            w[120] = q[104]
            w[121] = (q[124]*q[120]*q[119]+q[106]*q[119]-q[124]*q[120]*q[119]*q[106]*q[119])
            w[122] = q[103]*(1-q[116])
            w[123] = ((q[106]+q[104]-q[106]*q[104])+q[128]-(q[106]+q[104]-q[106]*q[104])*q[128])
            w[124] = q[110]
            w[125] = (q[110]+q[106]-q[110]*q[106])
            w[126] = (q[110]+q[106]-q[110]*q[106])
            w[127] = (q[126]+q[125]-q[126]*q[125])
            w[128] = q[127]
            w[129] = q[127]*q[125]
            w[130] = q[129]
            w[131] = q[119]
            w[132] = (p[24]+q[59]-p[24]*q[59])*(p[23]+q[58]-p[23]*q[58])*(p[17]+q[21]-p[17]*q[21])*q[108]*(p[5]+p[6]-p[5]*p[6])*q[145]*q[113]*q[112]*q[105]*p[34]*q[151]*(1-q[135])*(1-q[137])*(1-q[134])
            w[133] = q[132]*q[105]*q[112]*(1-q[134])
            w[134] = (p[24]+q[59]-p[24]*q[59])*(p[12]+p[13]-p[12]*p[13])*q[108]*q[146]*q[111]*q[112]*(1-q[132])*(1-q[138])*(1-q[133])*(1-q[142])
            w[135] = q[134]*(1-q[132])*(1-q[133])
            w[136] = ((p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[112]*q[111]*q[105]*q[108]+p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[137]*q[115]-p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[112]*q[111]*q[105]*q[108]*p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[137]*q[115])+p[14]*q[138]-(p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[112]*q[111]*q[105]*q[108]+p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[137]*q[115]-p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[112]*q[111]*q[105]*q[108]*p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[137]*q[115])*p[14]*q[138])*(1-q[133])*(1-q[154])*(1-(p[21]+q[57]-p[21]*q[57]))
            w[137] = p[14]*q[136]
            w[138] = q[136]
            w[139] = (((p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[105]*q[145]*p[34]+p[22]*p[14]*q[105]*q[145]*q[151]-(p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[105]*q[145]*p[34]*p[22]*p[14]*q[105]*q[145]*q[151])+q[154]-((p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[105]*q[145]*p[34]+p[22]*p[14]*q[105]*q[145]*q[151]-(p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[105]*q[145]*p[34]*p[22]*p[14]*q[105]*q[145]*q[151])*q[154])*(1-q[132])*(1-q[136])*(1-q[134])
            w[140] = (p[22]*q[139]+(p[21]+q[57]-p[21]*q[57])*q[142]-p[22]*q[139]*(p[21]+q[57]-p[21]*q[57])*q[142])*(1-q[133])*(1-q[135])*(1-q[137])
            w[141] = q[139]
            w[142] = (p[21]+q[57]-p[21]*q[57])*p[22]*q[105]*q[145]*(1-q[139])*(1-q[132])*(1-q[134])
            w[143] = q[142]
            w[144] = q[142]
            w[145] = (q[109]*q[114]+q[109]*q[151]-q[109]*q[114]*q[109]*q[151])*(1-q[148])*(1-p[38])
            w[146] = (q[109]*q[148]+q[109]*(p[12]+p[13]-p[12]*p[13])-q[109]*q[148]*q[109]*(p[12]+p[13]-p[12]*p[13]))
            w[147] = q[114]*q[153]
            w[148] = (((((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))+q[136]-((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))*q[136])+q[142]-(((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))+q[136]-((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))*q[136])*q[142])+p[37]-((((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))+q[136]-((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))*q[136])+q[142]-(((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))+q[136]-((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))*q[136])*q[142])*p[37])
            w[149] = (q[145]*p[26]+q[154]*p[26]-q[145]*p[26]*q[154]*p[26])*(1-q[153])*(1-q[142])
            w[150] = p[31]
            w[151] = q[150]
            w[152] = q[148]*p[33]
            w[153] = q[149]*(1-q[152])
            w[154] = (1-p[36])*q[114]
            dqdt = act_f(af, x=w, b=p[0], u=threshold_values) - decay_rates*q
            return dqdt

        return system_ode

    def solve(self, t_span=(0, 55), t_eval=None, grinitial_conditions=None, grmethod=None, grout_of_the_network_arguments_eval=None, af=None, decay_rates_list=None, grthreshold_values=None):
        if grinitial_conditions is None:
            grinitial_conditions = self.initial_conditions
        if grmethod is None:
            grmethod = 'LSODA'
        if grout_of_the_network_arguments_eval is None:
            grout_of_the_network_arguments_eval = self.initial_out
        if decay_rates_list is None:
            decay_rates_list = np.full(self.n, 1)
        if af is None:
            af = 'sigmoid'
        if grthreshold_values is None:
            grthreshold_values = np.full(self.n, 0.5)

        ode_system = self.create_ode_system(af, decay_rates_list, grthreshold_values)
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)

        if self.out_of_network_arguments is None:
            sol = solve_ivp(
                ode_system,
                t_span,
                grinitial_conditions,
                method=grmethod,
                t_eval=t_eval,
                dense_output=True
            )
        else:
            sol = solve_ivp(
                ode_system,
                t_span,
                grinitial_conditions,
                args=grout_of_the_network_arguments_eval,
                method=grmethod,
                t_eval=t_eval,
                dense_output=True
            )

        results = {
            'time': sol.t,
            'nodes': {}
        }
        for i, node in enumerate(self.array_of_elements):
            results['nodes'][node] = {
                'values': sol.y[i],
                'initial': sol.y[i, 0],
                'final': sol.y[i, -1],
                'max': np.max(sol.y[i]),
                'min': np.min(sol.y[i]),
                'mean': np.mean(sol.y[i])
            }

        return results
import re
import os
import tempfile
class BooleanToFuzzyProbabilistic:
    def __init__(self):
        self.ref_q = {}
        self.ref_p = {}
        self.log_dir = "./logs_conversion"
        os.makedirs(self.log_dir, exist_ok=True)
    def process_file(self, file=None, file_nodes=None, file_inputs=None):
        if None in (file, file_nodes, file_inputs):
            return None, "No file"
        try:
            fd_probabilistic, path_probabilistic = tempfile.mkstemp(suffix='.txt', prefix='probabilistic_')
            with open(file.name, 'r', encoding='utf-8') as f_in, \
                 os.fdopen(fd_probabilistic, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    line = line.strip()
                    if line:
                        f_out.write(self.convert_line(line) + '\n')
            return path_probabilistic, 'Success'
        except Exception as e:
            return None, f"Error: {str(e)}"
    def convert_boolean_expression(self, expr):
        if '!' in expr or '|' in expr or '&' in expr:
            if match := re.search(r'!(\w+)', expr):
                return self.convert_boolean_expression(
                    expr.replace(f'!{match.group(1)}', f'(1-{match.group(1)})')
                )
            if match := re.search(r'([^&|]+)&([^&|]+)', expr):
                return self.convert_boolean_expression(
                    expr.replace(match.group(0), f'{match.group(1)}*{match.group(2)}')
                )
            if match := re.search(r'([^&|]+)\|([^&|]+)', expr):
                return self.convert_boolean_expression(
                    expr.replace(match.group(0), f'({match.group(1)}+{match.group(2)}-{match.group(1)}*{match.group(2)})')
                )
        return expr
    def convert_line(self, boolean_expression):
        boolean_expression = boolean_expression.replace(' ', '')
        if '(' not in boolean_expression:
            return self.convert_boolean_expression(boolean_expression)
        markers = {}
        i = 0
        while '(' in boolean_expression:
            if match := re.search(r'\(([^()]+)\)', boolean_expression):
                marker = self.marker(i)
                markers[marker] = match.group(0)
                boolean_expression = boolean_expression.replace(match.group(0), marker)
                i += 1
            else:
                break
        boolean_expression = self.convert_boolean_expression(boolean_expression)
        for marker, parenthesis_expr in reversed(markers.items()):
            inner = parenthesis_expr.strip('()')
            converted_inner = self.convert_boolean_expression(inner)
            boolean_expression = boolean_expression.replace(marker, converted_inner)

        return boolean_expression
    def marker(self,marker_number :int=0) -> str:
        return f"_A{marker_number}_"
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
@dataclass
class PlotConfig:
    variables: List[str]
    labels: List[str]
    colors: List[str]
    styles: List[str]
    legfontsize: float
    legncol: int
    title: str

SUBNETWORKS = {
    'Macrophage inputs': PlotConfig(
        variables=[],
        labels=[],
        colors=[],
        styles=[],
        legfontsize=12,
        legncol=1,
        title='Microenviroment'
    ),
    'Macrophage Metabolism': PlotConfig(
        variables=['MGlycolysis', 'MOXPHOS', 'MAMPATPratio', 'MAMPK'],
        labels=['Glycolysis', 'OXPHOS', 'AMP/ATP', 'AMPK'],
        colors=['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red'],
        styles=['-', ':', '--', '-.'],
        legfontsize=12,
        legncol=2,
        title='Metabolism'
    ),
    'Phagocytosis': PlotConfig(
        variables=['PHAGOSOME','PHAGOCYTOSIS', 'PROCESSING', 'MHC2'],
        labels=['Endocytosis','Phagocytic program', 'Antigen Processing', 'MHC II-CD80/86'],
        colors=['#FFD700', '#32CD32', '#1E90FF', '#FF6347'],
        styles=['-', '-', '-', '-'],
        legfontsize=12,
        legncol=1,
        title='Phagocytosis'
    ),
    'Macrophage transcription factors': PlotConfig(
        variables=['MNFKB', 'MAP1', 'IRF3', 'IRF4', 'IRF7'],
        labels=['NF-κB', 'AP-1', 'IRF3', 'IRF4', 'IRF7'],
        colors=['#6A5ACD', '#FF69B4', '#00FA9A', '#FF4500', '#9932CC'],
        styles=['-.', '-', '-', '-', '-'],
        legfontsize=12,
        legncol=2,
        title='Transcription factors'
    ),
    'Secreted cytokines': PlotConfig(
        variables=['IL6OUT', 'IL12OUT', 'IL18OUT', 'IL33OUT', 'IL10OUT','IFNA','IL1BOUT', 'TNFAOUT'],
        labels=['IL-6', 'IL-12', 'IL-18', 'IL-33', 'IL-10','IFN-α','IL-1β','TNF-α'],
        colors=['brown','tab:brown', 'tab:cyan', 'tab:red', 'orange', 'blueviolet', '#3CB371', '#FF6347', '#000000'],
        styles=['-.',':', ':', ':', '-.', ':', '-', ':', '-'],
        legfontsize=12,
        legncol=2,
        title='Secreted cytokines'
    ),
    'M1': PlotConfig(
        variables=['IFNA', 'IL1BOUT', 'IL6OUT', 'IL12OUT', 'IL33OUT', 'IL18OUT', 'TNFAOUT'],
        labels=['IFN-α', 'IL-1β', 'IL6', 'IL12', 'IL33', 'IL18', 'TNF-α'],
        colors=['#3CB371', '#FF6347','tab:brown','tab:cyan', 'orange', 'tab:red', '#000000'],
        styles=['-', ':', ':', ':', '-.', ':', '-'],
        legfontsize=11,
        legncol=2,
        title='M1'
    ),
    'M2a': PlotConfig(
        variables=['PPARG', 'IL10OUT', 'JMJD3', 'STAT6'],
        labels=['PPAR-γ', 'IL10', 'JMJD3', 'STAT6'],
        colors=['#DAA520','blueviolet', '#20B2AA', '#CD5C5C'],
        styles=['-', ':', '-', '-'],
        legfontsize=12,
        legncol=2,
        title='M2a'
    ),
    'M2b': PlotConfig(
        variables=['ERK', 'IL10OUT' ],
        labels=['ERK', 'IL10'],
        colors=['#B8860B', 'blueviolet'],
        styles=['-', ':'],
        legfontsize=12,
        legncol=1,
        title='M2b'
    ),
    'M2c': PlotConfig(
        variables=['STAT3', 'IL10OUT'],
        labels=['STAT3', 'IL10'],
        colors=['#DB7093','blueviolet'],
        styles=['-', ':'],
        legfontsize=12,
        legncol=1,
        title='M2c'
    ),
    'Lymphocyte phenotype': PlotConfig(
        variables=['TBET', 'GATA3', 'RORGT', 'FOXP3', 'BCL6'],
        labels=['Th1', 'Th2', 'Th17', 'Treg', 'Tfh'],
        colors=['#228B22', '#4B0082', '#008080', '#B22222', '#A0522D'],
        styles=['-', '-', '-', '-', '-'],
        legfontsize=12,
        legncol=1,
        title='Lymphocyte phenotype'
    ),
    'Lymphocyte inputs': PlotConfig(
        variables=[],
        labels=[],
        colors=[],
        styles=[],
        legfontsize=12,
        legncol=1,
        title='Microenviroment'
    ),
    'Antigen presentation': PlotConfig(
        variables=['TCR', 'CD28', 'CTLA4', 'NDRG1'],
        labels=['TCR', 'CD28', 'CTLA4', 'NDRG1'],
        colors=['#7B68EE', '#48D1CC', '#FFA07A', '#F08080'],
        styles=['-', '-', '-', '-'],
        legfontsize=12,
        legncol=2,
        title='Antigen presentation'
    ),
    'T Metabolism': PlotConfig(
        variables=['Glycolysis', 'OXPHOS', 'AMPATPratio', 'AMPK'],
        labels=['Glycolysis', 'OXPHOS', 'AMP/ATP', 'AMPK'],
        colors=['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red'],
        styles=['-', ':', '--', '-.'],
        legfontsize=12,
        legncol=2,
        title='Metabolism'
    ),
    'T Transcription factors': PlotConfig(
        variables=['AP1', 'NFAT', 'NFKB'],
        labels=['AP1', 'NFAT', 'NFKB'],
        colors=['#FF69B4', '#FFB6C1', 'black'],
        styles=['-', '-', '-.'],
        legfontsize=12,
        legncol=2,
        title='Transcription factors'
    ),
    'Activation markers': PlotConfig(
        variables=['IL2G', 'MTORC1', 'MTORC2'],
        labels=['IL2G', 'MTORC1', 'MTORC2'],
        colors=['#FA8072', '#F4A460', '#C71585'],
        styles=['-', '-', '-'],
        legfontsize=12,
        legncol=2,
        title='Activation markers'
    ),
    'Th1': PlotConfig(
        variables=['TBET', 'IFNG'],
        labels=['TBET', 'IFN-γ'],
        colors=['#228B22', '#808000'],
        styles=['-', '-'],
        legfontsize=12,
        legncol=1,
        title='Th1'
    ),
    'Th2': PlotConfig(
        variables=['GATA3', 'IL4'],
        labels=['GATA3', 'IL4'],
        colors=['#4B0082', '#556B2F'],
        styles=['-', ':'],
        legfontsize=12,
        legncol=1,
        title='Th2'
    ),
    'Th17': PlotConfig(
        variables=['IL17', 'IL21', 'RORGT'],
        labels=['IL17', 'II21', 'RORGT'],
        colors=['#FF8C00', '#E9967A', '#008080'],
        styles=[':', ':', '-'],
        legfontsize=12,
        legncol=2,
        title='Th17'
    ),
    'Treg': PlotConfig(
        variables=['FOXP3', 'TGFB', 'CTLA4DIM','IL10OUT'],
        labels=['FOXP3', 'TGFB', 'CTLA4DIM','IL10'],
        colors=['#B22222', '#DAA520', '#BDB76B', '#9370DB'],
        styles=['-', '-', '-', ':'],
        legfontsize=12,
        legncol=2,
        title='Treg'
    ),
    'Tfh': PlotConfig(
        variables=['BCL6', 'IL21', 'CD40L', 'IL9'],
        labels=['BCL6', 'IL21', 'CD40L', 'IL9'],
        colors=['#A0522D', '#E9967A', '#66CDAA', '#8FBC8F'],
        styles=['-', ':', '-', ':'],
        legfontsize=12,
        legncol=2,
        title='Tfh'
    )
}
import matplotlib.pyplot as plt
from math import ceil
import gradio as gr
import plotly.graph_objects as go
class NetworkInterface(NetworkEquations):
    DEFAULT_MIN_SLIDER = 0.0
    DEFAULT_MAX_SLIDER = 1.0
    DEFAULT_SLIDER_STEP = 0.01
    DEFAULT_SLIDER_VALUE = 0.5

    #
    DEFAULT_TIME_SPAN = (0, 55)
    DEFAULT_DECAY_RATE = 1.0
    DEFAULT_THRESHOLD_VALUE = 0.5
    MAX_NODES_TO_PLOT = 16

    #
    PLOT_Y_RANGE = (-0.1, 1.1)
    PLOT_HEIGHT = 500
    TITLE_Y_POSITION = 0.95
    TITLE_X_POSITION = 0.5

    #
    SUBPLOT_COLUMNS = 5
    SUBPLOT_FIGURE_WIDTH = 21
    SUBPLOT_HEIGHT_MULTIPLIER = 6.5
    SUBPLOT_TIGHT_PAD = 0.8
    SUBPLOT_TIGHT_PAD_Y = 0.0

    #
    LEGEND_BBOX_X = 0.0
    LEGEND_BBOX_Y = 1.5
    LEGEND_BBOX_WIDTH = 1.0
    LEGEND_BBOX_HEIGHT = 0.02
    LEGEND_FONT_SIZE = 14
    LEGEND_TITLE_FONT_SIZE = 14

    #
    BAR_ALPHA = 0.85
    BAR_COLOR = 'purple'
    BAR_Y_LIMIT = (-0.01, 1.1)

    #
    GAUSSIAN_STD = 0.15
    GAUSSIAN_MEAN_0 = 0
    GAUSSIAN_MEAN_05 = 0.5
    GAUSSIAN_MEAN_1 = 1

    #
    INPUT_GROUPS = {
        'Antigen microenviroment': sorted(['LPSE', 'LP', 'DSRNA', 'SSRNA', 'CPGDNA',
                                           'GMCSFE', 'IC3B', 'IGGC', 'IGAC', 'CITDSRNA', 'CITSSRNA']),
        'Cytokine microenviroment': ['IFNGE', 'IFNGL', 'IL1BE', 'IL4E', 'IL4L', 'IL6E', 'IL10E', 'IL10L', 'IL12E', 'IL18E', 'IL21E', 'IL33E', 'TGFBE','TNFA'],
        'Activation parameters': sorted(['b', 'TCD8086', 'TMHC2', 'MHC2E', 'CD8086E']),
        'Macrophage nutrients': sorted(['MO2', 'MGLC']),
        'Nutrients and molecular therapies': sorted(['GLC', 'O2', 'CIAP', 'GLN', 'MFA', 'FA',
                                                      'TRP', 'METF', 'RAPA', 'PRED', 'CS'])}

    #
    HIDDEN_SLIDERS = ['IL10OUT', 'IL6OUT', 'IL33OUT', 'IL18OUT', 'IL12OUT']

    #
    DEFAULT_PLOT_OPTIONS = ['OXPHOS', 'IL10', 'TGFB', 'RORGT', 'IL21', 'IL17', 'BCL6']

    #
    MAX_OUT_SLIDERS_DEFAULT = [100]  #
    ACTIVATION_FUNCTIONS = ['sigmoid', 'tanh', 'hill']
    DEFAULT_ACTIVATION = 'sigmoid'
    ODE_SOLVERS = ['DOP853', 'LSODA']
    DEFAULT_SOLVER = 'LSODA'
    DISTRIBUTION_TYPES = ['normal', 'uniform', 'laplace']
    DEFAULT_DISTRIBUTION = 'uniform'
    def __init__(self, features: dict):
        super().__init__()
        self.value_sliders = self.initial_conditions
        self.max_sliders_out = [100] + (len(self.out_of_network_arguments) - 3) * [1] + 2 * [100]
        self.value_sliders_out = self.initial_out
        self.distribution_type = None
        self.features = features
        self.subnetwork_options = list(features.keys())
        self.last_results = None
        self.subnetwork_name = None
        self.last_fig2 = None
        self.last_fig3 = None
        self.gradio_interface()
    def create_bar_chart(self, axi, labels, labels_mute, values, title,
                      bbox_position, ncol=1, y_limit=None):
        axi.set_xticks(range(len(labels)))
        axi.set_xticklabels(labels_mute if labels_mute else labels,
                            rotation=90, ha='center', fontsize=13)
        axi.tick_params(axis='both', labelsize=13)
        axi.bar(labels, values, color=self.BAR_COLOR, alpha=self.BAR_ALPHA, label='_nolegend_')
        axi.legend(bbox_to_anchor=bbox_position, loc=1, ncol=ncol,
                  mode="expand", borderaxespad=0.,title=title, title_fontsize='xx-large')
        if y_limit:
            axi.set_ylim(*y_limit)
        return axi

    def create_macrophage_inputs_chart(self, axi, bars_reference, subr, bbox_position):
        labels = ['TGFBE', 'IFNGE', 'IL10E', 'IL4E', 'IL1BE', 'TNFA']
        labels_mute = ['TGFB', 'IFNG', 'IL10', 'IL4E', 'IL1B', 'TNFA']
        values = [bars_reference[label] for label in labels]
        return self.create_bar_chart(axi, labels, labels_mute, values, subr,
                                       bbox_position, ncol=1, y_limit=self.BAR_Y_LIMIT)

    def create_lymphocyte_inputs_chart(self, axi, bars_reference, subr, bbox_position):
        reference = ['IL12E', 'IL18E', 'IL33E', 'IL6E', 'IL10L', 'IL4L','IL21E']
        labels = ['TCR/CD28', 'IL12L', 'IL18L', 'IL33L', 'IL6L', 'IL10L', 'IL4L','IL21L']
        TCR_value = 1 if bars_reference.get('MHC2E') == 1 and bars_reference.get('CD8086E') else 0
        values = [bars_reference[ref] for ref in reference]
        values = [TCR_value] + values
        return self.create_bar_chart(axi, labels, None, values, subr,
                                       bbox_position, ncol=1, y_limit=self.BAR_Y_LIMIT)

    def create_lymphocyte_phenotypes_chart(self, axi, results, subr):
        config = self.features[subr]
        labels = config.labels
        axi.set_xticks(range(len(labels)))
        axi.set_xticklabels(labels, rotation=90, ha='right', fontsize=self.LEGEND_FONT_SIZE)
        phenotype_population = [results['nodes'][k]['max'] for k in config.variables
                               if k in results['nodes']]
        axi.bar(labels, phenotype_population, color=self.BAR_COLOR)
        axi.legend(bbox_to_anchor=(self.LEGEND_BBOX_X, 1.3, self.LEGEND_BBOX_WIDTH, 0.09),
                  loc=1, ncol=1, mode="expand", borderaxespad=0.,
                  prop={'size': self.LEGEND_FONT_SIZE}, title=config.title,
                  title_fontsize='xx-large')
        axi.tick_params(axis='both', labelsize=self.LEGEND_FONT_SIZE)
        axi.set_ylim(*self.BAR_Y_LIMIT)
        return axi

    def create_line_plot(self, axi, results, subr, bbox_position, ncol):
        config = self.features[subr]
        for node, col, sty, label in zip(config.variables, config.colors,
                                         config.styles, config.labels):
            data = results['nodes'][node]
            axi.plot(results['time'], data['values'], label=label,
                    color=col, linestyle=sty, linewidth=2)
            axi.legend(bbox_to_anchor=bbox_position, loc=1, ncol=ncol,
                      mode="expand", borderaxespad=0., prop={'size': self.LEGEND_FONT_SIZE},
                      title=config.title,  title_fontsize='xx-large')
            axi.tick_params(axis='both', labelsize=14)
            axi.set_xlabel('Time', fontsize=14)
            axi.set_ylabel('Expression', fontsize=14)
        axi.set_ylim(*self.PLOT_Y_RANGE)
        axi.grid(False)
        return axi
    def design_and_personalize_subplots(self, results, inputs, start_idx, end_idx):
        n_rows = ceil(len(list(self.features.keys())[start_idx:end_idx]) / self.SUBPLOT_COLUMNS)
        fig2, axis = plt.subplots(n_rows, self.SUBPLOT_COLUMNS, figsize=(self.SUBPLOT_FIGURE_WIDTH, self.SUBPLOT_HEIGHT_MULTIPLIER * n_rows), layout="constrained")
        axis = axis.flatten()
        sub_names = list(self.features.keys())[start_idx:end_idx]
        positions_bbox_to_anchor = {subr: (self.LEGEND_BBOX_X, self.LEGEND_BBOX_Y, self.LEGEND_BBOX_WIDTH, self.LEGEND_BBOX_HEIGHT)
                                     for subr in self.features}
        bars_reference = dict(zip(self.out_of_network_arguments, inputs))
        for idx, (axi, subr) in enumerate(zip(axis, sub_names)):
            config=self.features[subr]
            if subr == 'Macrophage inputs':
                self.create_macrophage_inputs_chart(axi, bars_reference,config.title, positions_bbox_to_anchor[subr])
            elif subr == 'Lymphocyte inputs':
                self.create_lymphocyte_inputs_chart(axi, bars_reference,config.title, positions_bbox_to_anchor[subr])
            elif subr == 'Lymphocyte phenotype':
                self.create_lymphocyte_phenotypes_chart(axi, results, subr)
            else:
                self.create_line_plot(axi, results, subr, positions_bbox_to_anchor[subr], config.legncol)
        for j in range(idx + 1, len(axis)):
            axis[j].set_visible(False)
        plt.show()
        fig2.get_layout_engine().set(h_pad=self.SUBPLOT_TIGHT_PAD)
        return fig2

    def _create_slider(self, label: str, minimum: float = DEFAULT_MIN_SLIDER,
                       maximum: float = DEFAULT_MAX_SLIDER,
                       value: float = DEFAULT_SLIDER_VALUE,
                       step: float = DEFAULT_SLIDER_STEP,
                       visible: bool = True) -> gr.Slider:
        return gr.Slider(
            minimum=minimum,
            maximum=maximum,
            step=step,
            label=label,
            value=value,
            interactive=True,
            visible=visible
        )

    def create_slider_group(self, labels: List[str], values: List[float],
                             minimums, maximums, exceptions: Optional[List[str]] = None,
                             n_columns: int = 5) -> List[gr.Slider]:
        sliders = []
        n_rows = (len(labels) + n_columns - 1) // n_columns
        for row in range(n_rows):
            with gr.Row(variant="compact"):
                start_idx = row * n_columns
                end_idx = min((row + 1) * n_columns, len(labels))
                for idx in range(start_idx, end_idx):
                    visibility = exceptions is None or labels[idx] not in exceptions
                    slider = self._create_slider(
                        label=labels[idx],
                        minimum=minimums[idx] if isinstance(minimums, list) else minimums,
                        maximum=maximums[idx] if isinstance(maximums, list) else maximums,
                        value=values[idx],
                        visible=visibility
                    )
                    sliders.append(slider)
        return sliders

    def Inputs_tab(self, tab_title='Inputs', n_columns_in_tab: int = 5):
        with gr.Tab(tab_title):
            all_sliders_out = len(self.out_of_network_arguments) * [0]

            with gr.Row():
                with gr.Column():
                    solve_btn_in = gr.Button("Simulate network dynamics", variant="primary")
                with gr.Column():
                    reset_network_params = gr.Button('Reset input values', variant='secondary')

            with gr.Column():
                max_out_sliders_reference = dict(zip(self.out_of_network_arguments, self.max_sliders_out))
                value_reference = dict(zip(self.out_of_network_arguments, self.value_sliders_out))

                all_sliders_input = []
                all_sliders_out = len(self.out_of_network_arguments) * [0]

                for group_name, group_labels in self.INPUT_GROUPS.items():
                    gr.Markdown(f'{group_name}')
                    group_values = [value_reference[elemento] for elemento in group_labels]
                    group_minimus = len(group_labels) * [0]
                    group_max = [max_out_sliders_reference[key] for key in group_labels]
                    all_sliders_input += self.create_slider_group(group_labels, group_values,
                                                                    group_minimus, group_max,
                                                                    exceptions=None, n_columns=5)

                common_ref = dict(zip(self.out_of_network_arguments.tolist(), range(len(self.out_of_network_arguments))))
                for element in all_sliders_input:
                    old_idx = common_ref[element.label]
                    all_sliders_out[old_idx] = element

                for name, element in zip(self.out_of_network_names, all_sliders_out):
                    element.label = name

                with gr.Row():
                    reset_decay_rates_sliders = gr.Button('Reset decay rates', variant='secondary')

                labels_dr = [f'Decay rate: ({par})' for par in self.array_of_elements]
                values_dr = [self.DEFAULT_DECAY_RATE for _ in range(self.n)]
                all_sliders_alpha = self.create_slider_group(labels_dr, values_dr,
                                                                minimums=0, maximums=10,
                                                                exceptions=None, n_columns=5)

        return solve_btn_in, reset_network_params, all_sliders_out, all_sliders_input, all_sliders_alpha, reset_decay_rates_sliders

    def Initial_conditions_tab(self):
        with gr.Tab("Initial conditions"):
            solve_btn_ic = gr.Button("Simulate network dynamics", variant="primary")
            with gr.Row():
                with gr.Column():
                    restart_initial_conditions = gr.Button('Reset initial conditions', variant='secondary')
                with gr.Column():
                    set_gaussian_ic = gr.Dropdown(choices=['0', '0.5', '1'],
                                                  label='Gaussian distribution for initial conditions',
                                                  value='0')

            nodes_names = self.array_of_elements.tolist()
            nodes_values = self.value_sliders.tolist()
            all_sliders_ic = self.create_slider_group(labels=nodes_names, values=nodes_values,
                                                         minimums=self.DEFAULT_MIN_SLIDER,
                                                         maximums=self.DEFAULT_MAX_SLIDER,
                                                         exceptions=self.HIDDEN_SLIDERS)

        return solve_btn_ic, restart_initial_conditions, set_gaussian_ic, all_sliders_ic

    def Thresholds_tab(self):
        with gr.Tab("Threshold values"):
            with gr.Row():
                with gr.Column():
                    solve_btn_tv = gr.Button("Simulate network dynamics", variant="primary")
                with gr.Column():
                    reset_threshold_values = gr.Button('Reset threshold values', variant='secondary')

            labels_um = [f'Threshold U ({par})' for par in self.array_of_elements]
            values_um = [self.DEFAULT_THRESHOLD_VALUE for _ in range(self.n)]
            all_sliders_threshold = self.create_slider_group(labels_um, values_um,
                                                            minimums=self.DEFAULT_MIN_SLIDER,
                                                            maximums=self.DEFAULT_MAX_SLIDER,
                                                            exceptions=None, n_columns=5)

        return solve_btn_tv, reset_threshold_values, all_sliders_threshold
    def Results_tab(self):
        with gr.Tab("Results"):
            solve_btn_re = gr.Button("Simulate network dynamics", variant="primary")
            with gr.Row():
                GPlot = gr.Plot()
            with gr.Row():
                LymphoPlot = gr.Plot()
            with gr.Row():
                with gr.Column():
                    download_btn_fig2 = gr.Button("Download Figure 2 (Subplots 0-10) as JPG", variant="secondary")
                    download_fig2 = gr.File(label="Download Fig2", visible=True)
                with gr.Column():
                    download_btn_fig3 = gr.Button("Download Figure 3 (Subplots 10-20) as JPG", variant="secondary")
                    download_fig3 = gr.File(label="Download Fig3", visible=True)
        return solve_btn_re, GPlot, LymphoPlot, download_btn_fig2, download_fig2, download_btn_fig3, download_fig3


    def Boolean_to_Fuzzy_tab(self):
        with gr.Tab("Convert and process your own boolean networks"):
            with gr.Row():
                procesar_btn = gr.Button("Process file", variant="primary", size="lg")
                limpiar_btn = gr.Button("Delete file", variant="secondary")
            with gr.Row():
                with gr.Column():
                    archivo_boolean = gr.File(
                        label="📄 Boolean rules file",
                        file_types=[".txt"]
                    )
                with gr.Column():
                    archivo_nodes = gr.File(
                        label="📄 Nodes file",
                        file_types=[".txt"]
                    )
                with gr.Column():
                    archivo_inputs = gr.File(
                        label="📄 Inputs file",
                        file_types=[".txt"]
                    )
            with gr.Row():
                with gr.Column():
                    archivo_out_boolean = gr.File(
                        label="📥 Probablistic expressions file",
                        interactive=False
                    )
            info_text = gr.Textbox(label="ℹ️ Information", lines=3)
        return procesar_btn, limpiar_btn, archivo_boolean, archivo_nodes, archivo_inputs, archivo_out_boolean, info_text

    def Plot_custom_tab(self):
        with gr.Tab("Plot customization"):
            with gr.Row():
                solve_btn_pc = gr.Button("Simulate network dynamics", variant="primary")
            with gr.Row():
                select_box = gr.CheckboxGroup(
                    choices=list(self.array_of_elements),
                    value=['OXPHOS', 'AMPK', 'Glycolysis'],
                    label="Choose the nodes to plot (15 nodes max)"
                )
            with gr.Row():
                figura = gr.Plot()
            return solve_btn_pc, select_box, figura
    def save_matplotlib_as_jpg(self, fig: plt.Figure, filename: str = "plot.jpg") -> str:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        fig.savefig(temp_file.name, format='jpg', dpi=300, bbox_inches='tight')
        return temp_file.name
    def gradio_interface(self, n_columns: int = 5):
        with gr.Blocks(title="Network simulator") as demo:
            activation = gr.Dropdown(
                choices=self.ACTIVATION_FUNCTIONS,
                label="Activation function",
                value=self.DEFAULT_ACTIVATION,
                render=False
            )
            method = gr.Dropdown(
                choices=self.ODE_SOLVERS,
                label="ODE Solver",
                value=self.DEFAULT_SOLVER,
                render=False
            )
            distribution = gr.Dropdown(
                choices=self.DISTRIBUTION_TYPES,
                value=self.DEFAULT_DISTRIBUTION,
                label='Distribution to simulate network dynamics',
                render=False
            )

            with gr.Tabs():
                solve_btn_in, reset_network_params, all_sliders_out, all_sliders_input, all_sliders_alpha, reset_decay_rates_sliders = self.Inputs_tab('Inputs', n_columns)
                solve_btn_ic, restart_initial_conditions, set_gaussian_ic, all_sliders_ic = self.Initial_conditions_tab()
                solve_btn_tv, reset_threshold_values, all_sliders_threshold = self.Thresholds_tab()
                solve_btn_re, GPlot, LymphoPlot, download_btn_fig2, download_fig2, download_btn_fig3, download_fig3= self.Results_tab()
                procesar_btn, limpiar_btn, archivo_boolean, archivo_nodes, archivo_inputs, archivo_out_boolean, info_text = self.Boolean_to_Fuzzy_tab()
                solve_btn_pc, select_box, figura = self.Plot_custom_tab()

                with gr.Tab("Additional settings"):
                    with gr.Row(variant="compact"):
                        activation.render()
                        method.render()
                    with gr.Row(variant="compact"):
                        distribution.render()

            variables_to_control = all_sliders_ic + all_sliders_out + all_sliders_alpha + all_sliders_threshold + [activation] + [method] + [select_box]

            set_gaussian_ic.change(
                fn=self.set_gaussian_conditions,
                inputs=set_gaussian_ic,
                outputs=all_sliders_ic
            )
            reset_network_params.click(
                fn=self.reset_network_parameters,
                outputs=all_sliders_out
            )
            distribution.change(
                fn=self.set_distribution,
                inputs=distribution,
                outputs=all_sliders_ic
            )
            restart_initial_conditions.click(
                fn=self.restart_sliders_value,
                outputs=all_sliders_ic
            )
            reset_decay_rates_sliders.click(
                fn=self.set_decay_rates,
                outputs=all_sliders_alpha
            )
            reset_threshold_values.click(
                fn=self.reset_threshold_values,
                outputs=all_sliders_threshold
            )
            select_box.change(
                fn=self.update_general_plot,
                inputs=select_box,
                outputs=figura
            )

            botones = [solve_btn_ic, solve_btn_in, solve_btn_tv, solve_btn_pc, solve_btn_re]
            for button in botones:
                button.click(
                    fn=self.after_push_button,
                    inputs=variables_to_control,
                    outputs=[figura, GPlot, LymphoPlot]
                )
            download_btn_fig2.click(
                fn=lambda: self.save_matplotlib_as_jpg(self.last_fig2, "subplots_0_10.jpg") if self.last_fig2 else None,
                outputs=download_fig2
            )
            download_btn_fig3.click(
                fn=lambda: self.save_matplotlib_as_jpg(self.last_fig3, "subplots_10_20.jpg") if self.last_fig3 else None,
                outputs=download_fig3
            )
            convertidor = BooleanToFuzzyProbabilistic()
            procesar_btn.click(
                fn=convertidor.process_file,
                inputs=[archivo_boolean, archivo_nodes, archivo_inputs],
                outputs=[archivo_out_boolean, info_text]
            )

        return demo.launch(share=False)
    def after_push_button(self, *values):
        nodes = values[:self.n]
        out_params = values[self.n:self.n + len(self.out_of_network_arguments)]
        decay_rates = values[self.n + len(self.out_of_network_arguments):self.n + len(self.out_of_network_arguments) + self.n]
        threshold_values = values[self.n + len(self.out_of_network_arguments) + self.n:
                                  self.n + len(self.out_of_network_arguments) + self.n + self.n]
        activation = values[self.n + self.n + self.n + len(self.out_of_network_arguments)]
        method = values[self.n + self.n + self.n + len(self.out_of_network_arguments) + 1]
        selection_box_plot = values[self.n + self.n + self.n + len(self.out_of_network_arguments) + 2]

        Resultados = self.solve(
            t_span=self.DEFAULT_TIME_SPAN,
            t_eval=None,
            grinitial_conditions=nodes,
            grmethod=method,
            grout_of_the_network_arguments_eval=out_params,
            af=activation,
            decay_rates_list=decay_rates,
            grthreshold_values=threshold_values
        )
        self.last_results = Resultados
        Fig1 = self.plot_results(Resultados, selection_box_plot)
        Fig2 = self.design_and_personalize_subplots(Resultados, out_params, start_idx=0, end_idx=10)
        Fig3 = self.design_and_personalize_subplots(Resultados, out_params, start_idx=10, end_idx=20)
        self.last_fig2 = Fig2
        self.last_fig3 = Fig3
        return Fig1, Fig2, Fig3

    def set_gaussian_conditions(self, val: str):
        if val == '0':
            return np.absolute(np.random.normal(self.GAUSSIAN_MEAN_0, self.GAUSSIAN_STD, self.n)).tolist()
        elif val == '0.5':
            return np.absolute(np.random.normal(self.GAUSSIAN_MEAN_05, self.GAUSSIAN_STD, self.n)).tolist()
        elif val == '1':
            return np.absolute(np.random.normal(self.GAUSSIAN_MEAN_1, self.GAUSSIAN_STD, self.n)).tolist()
    def update_general_plot(self, selected_nodes: list) -> go.Figure:
        if not hasattr(self, 'last_results') or self.last_results is None:
            return go.Figure().update_layout(title="Please run a simulation first to plot selected nodes.")
        return self.plot_results(self.last_results, selected_nodes)
    def set_distribution(self, option):
        if option == 'normal':
            return np.absolute(np.random.normal(self.GAUSSIAN_MEAN_05, self.GAUSSIAN_STD, self.n)).tolist()
        elif option == 'uniform':
            return np.absolute(np.random.uniform(self.GAUSSIAN_MEAN_05, self.GAUSSIAN_STD, self.n)).tolist()
        elif option == 'laplace':
            return np.absolute(np.random.laplace(self.GAUSSIAN_MEAN_05, self.GAUSSIAN_STD, self.n)).tolist()

    def restart_sliders_value(self, *values):
        return self.value_sliders.tolist()

    def reset_network_parameters(self, *values):
        return self.value_sliders_out.tolist()

    def reset_threshold_values(self, *values):
        return np.full(self.n, self.DEFAULT_THRESHOLD_VALUE).tolist()

    def set_decay_rates(self, *values):
        return np.ones(self.n).tolist()
    def create_Plot(self, external_inputs,s_idx,e_idx):
        if self.last_results is None:
            return go.Figure().update_layout(title='Run a network simulation first')
        else:
            Figura = self.design_and_personalize_subplots(self.last_results, external_inputs, s_idx, e_idx)
            return Figura
    def plot_results(self, results: dict = None, options: list = None) -> go.Figure:
        if options is None:
            options = self.DEFAULT_PLOT_OPTIONS

        if len(options) > self.MAX_NODES_TO_PLOT:
            return go.Figure().update_layout(title=f"Error: Change one of the nodes to add it to your graphic")

        fig = go.Figure()
        t = results['time']

        for option in options:
            fig.add_trace(go.Scatter(
                x=t, y=results['nodes'][option]['values'],
                mode='lines', name=option, line=dict(width=2),
                hovertemplate=f'{option}<br>Time: %{{x:.2f}}<br>Value: %{{y:.4f}}'
            ))

        title_config = {
            'text': "Nodes expression",
            'x': self.TITLE_X_POSITION,
            'y': self.TITLE_Y_POSITION,
            'xanchor': 'center',
            'yanchor': 'top'
        }

        fig.update_layout(
            title=title_config,
            yaxis=dict(range=self.PLOT_Y_RANGE, autorange=False),
            xaxis_title='Time',
            yaxis_title='Expression',
            hovermode='x unified',
            template='plotly_white',
            height=self.PLOT_HEIGHT
        )

        return fig
if __name__ == "__main__":
   network = NetworkInterface(SUBNETWORKS)
