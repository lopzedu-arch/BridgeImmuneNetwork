import matplotlib.pyplot as plt
import math
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

class NetworkEquations:
    def __init__(self):
        self.array_of_elements = None
        self.n = None
        self.out_of_network_arguments = None
        self.subnetwork_name = None
        self.Nodes()
        self.networkparameters()
        self.create_ode_system()
        self.solve()
    def Nodes(self):
        self.array_of_elements = np.array(['IFNGR', 'CSF2RA', 'IL1R', 'TLR4', 'FCGR', 'IL4RA',
                                           'IL10R', 'STAT1', 'STAT5', 'NFKB', 'PPARG', 'STAT6',
                                           'JMJD3', 'STAT3', 'SOCS3', 'IRF3', 'ERK', 'KLF4',
                                           'SOCS1', 'IRF4', 'IL1BOUT', 'IL12OUT', 'TAK123',
                                           'JNKMAPK', 'ERK12', 'MSK12', 'GSK3B', 'IL10OUT',
                                           'AKT1', 'AKT3', 'TLR2', 'TLR3', 'TLR7', 'TLR8', 'TLR9',
                                           'MYD88', 'IRAK4', 'IRAK12', 'TRAF6', 'TAB23TAK',
                                           'MKK36', 'P38', 'CREB1', 'MKK47', 'JNK', 'CJUN',
                                           'MKK12', 'CFOS', 'AP1', 'NEMOIKKB', 'RIP1', 'TLR4END',
                                           'TRIF', 'TRAF3', 'TBK1IKKI', 'IKKA', 'IRF7', 'IL6OUT',
                                           'IL18OUT', 'IL33OUT', 'IFNABOUT', 'PHAGOCYTOSIS',
                                           'PHAGOSOME', 'PROCESSING', 'MHC2', 'ITAM', 'SYK',
                                           'CARD9', 'VAV', 'RAC', 'CDC42', 'WASP', 'WAVE', 'CA',
                                           'LYN', 'FYN', 'DAP12', 'PI3K', 'PLC', 'AKT', 'PKC',
                                           'CD11BCD18', 'HCK', 'FGR', 'FCAR', 'MTOR', 'MTORC1',
                                           'MTORC2', 'LKB1', 'AMPK', 'Glycolysis', 'OXPHOS',
                                           'AMPATPratio', 'HIF1A', 'RIG1', 'MAVS', 'TRADD',
                                           'NEMOIKKAB', 'NEMOTBK1IKKE', 'TNFR1', 'TRAF2',
                                           'TNFAOUT', 'DECTIN1', 'SRC', 'DECTIN2', 'MR',
                                           'CLEC10A', 'MINCLE'])
        self.n = len(self.array_of_elements)
        return self.array_of_elements
    def networkparameters(self):
        """I think it is worth mentioning this method has to be optional
        for many networks without external parameters.However, we can adjust it later"""
        self.out_of_network_arguments = np.array(['b','LP', 'DSRNA', 'SSRNA', 'CPGDNA', 'IFNGE',
                                                  'GMCSFE', 'IL1BE', 'LPSE', 'IL4E', 'IL10E',
                                                  'IC3B', 'IGGC', 'IGAC', 'O2', 'FA', 'GLC',
                                                  'CITDSRNA', 'CITSSRNA', 'TNFA', 'CIAP'])

        return self.out_of_network_arguments
    def create_ode_system(self, af=None, decay_rates=None, threshold_values=None):
        """ the biggest changes are going to be implemented here. I underestimated the impact of the lambdify function on
        the performance of the solver. This new implementation should be way more efficient but less dynamic"""
        def act_f(a_function: str=None, x=None, b=None, u=None):
            if a_function is None:
                a_function = 'sigmoid'
            if a_function == 'sigmoid':
                return 1 / (1 + np.exp(-b*(x-u)))
            elif a_function == 'tanh':
                return (1+np.tanh(-b*(x-u)))/2
            elif a_function == 'relu':
                return np.maximum(0,-b*(x-u))
            else:
                raise ValueError(f"Invalid activation function: {a_function}. Choose from ['sigmoid', 'tanh', 'relu'].")
        """ right here  you have to copy the function you got with weigths_parsed.txt"""
        def system_ode(t,q,*p):
            w = np.zeros(len(q))
            w[0]=p[5]
            w[1]=p[6]
            w[2]=(p[7]+q[20]-p[7]*q[20])
            w[3]=p[8]
            w[4]=((p[12]+p[12]*p[7]-p[12]*p[12]*p[7])+q[107]-(p[12]+p[12]*p[7]-p[12]*p[12]*p[7])*q[107])
            w[5]=p[9]
            w[6]=(p[10]+q[27]-p[10]*q[27])
            w[7]=q[0]*(1-(q[18]+q[13]-q[18]*q[13]))
            w[8]=q[1]*(1-((q[13]+q[19]-q[13]*q[19])+q[18]-(q[13]+q[19]-q[13]*q[19])*q[18]))
            w[9]=(((((q[2]+q[49]-q[2]*q[49])+q[67]-(q[2]+q[49]-q[2]*q[49])*q[67])+q[79]-((q[2]+q[49]-q[2]*q[49])+q[67]-(q[2]+q[49]-q[2]*q[49])*q[67])*q[79])+q[80]-(((q[2]+q[49]-q[2]*q[49])+q[67]-(q[2]+q[49]-q[2]*q[49])*q[67])+q[79]-((q[2]+q[49]-q[2]*q[49])+q[67]-(q[2]+q[49]-q[2]*q[49])*q[67])*q[79])*q[80])+q[50]*q[100]*p[20]-((((q[2]+q[49]-q[2]*q[49])+q[67]-(q[2]+q[49]-q[2]*q[49])*q[67])+q[79]-((q[2]+q[49]-q[2]*q[49])+q[67]-(q[2]+q[49]-q[2]*q[49])*q[67])*q[79])+q[80]-(((q[2]+q[49]-q[2]*q[49])+q[67]-(q[2]+q[49]-q[2]*q[49])*q[67])+q[79]-((q[2]+q[49]-q[2]*q[49])+q[67]-(q[2]+q[49]-q[2]*q[49])*q[67])*q[79])*q[80])*q[50]*q[100]*p[20])*(1-(((q[13]+q[10]-q[13]*q[10])+q[17]-(q[13]+q[10]-q[13]*q[10])*q[17])+q[42]-((q[13]+q[10]-q[13]*q[10])+q[17]-(q[13]+q[10]-q[13]*q[10])*q[17])*q[42]))
            w[10]=q[5]
            w[11]=q[5]*(1-q[7])
            w[12]=q[5]
            w[13]=(q[6]+q[16]*q[41]-q[6]*q[16]*q[41])*(1-(q[10]+q[14]-q[10]*q[14]))
            w[14]=(q[9]+q[7]-q[9]*q[7])
            w[15]=(q[54]+q[98]-q[54]*q[98])
            w[16]=((q[4]+q[46]-q[4]*q[46])+q[77]-(q[4]+q[46]-q[4]*q[46])*q[77])
            w[17]=q[11]
            w[18]=(q[11]+q[60]-q[11]*q[60])
            w[19]=q[12]
            w[20]=(q[9]+q[9]*q[48]-q[9]*q[9]*q[48])
            w[21]=((q[7]+q[8]*q[9]-q[7]*q[8]*q[9])+q[48]*q[9]-(q[7]+q[8]*q[9]-q[7]*q[8]*q[9])*q[48]*q[9])
            w[22]=q[35]
            w[23]=q[22]
            w[24]=q[23]
            w[25]=(q[24]+q[41]-q[24]*q[41])
            w[26]=((1-q[29])+(1-q[28])-(1-q[29])*(1-q[28]))
            w[27]=((((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])+q[13]-((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])*q[13])+q[42]*q[48]-(((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])+q[13]-((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])*q[13])*q[42]*q[48])
            w[28]=q[77]*(1-q[0])
            w[29]=q[77]*(1-q[0])
            w[30]=p[1]
            w[31]=p[2]
            w[32]=p[3]
            w[33]=p[3]
            w[34]=p[4]
            w[35]=((((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])+q[33]-((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])*q[33])+q[34]-(((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])+q[33]-((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])*q[33])*q[34])
            w[36]=q[35]
            w[37]=q[36]
            w[38]=q[37]
            w[39]=(q[38]+q[50]-q[38]*q[50])
            w[40]=q[39]
            w[41]=q[40]
            w[42]=q[25]*(1-q[26])
            w[43]=q[39]
            w[44]=(q[43]+q[77]-q[43]*q[77])
            w[45]=q[44]
            w[46]=q[49]
            w[47]=q[16]
            w[48]=((q[45]+q[47]-q[45]*q[47])+q[25]*(1-q[26])-(q[45]+q[47]-q[45]*q[47])*q[25]*(1-q[26]))
            w[49]=q[39]
            w[50]=(q[52]+q[96]-q[52]*q[96])
            w[51]=q[3]
            w[52]=(q[31]+q[51]-q[31]*q[51])
            w[53]=((((q[52]+q[96]*q[95]-q[52]*q[96]*q[95])+q[37]*q[32]-(q[52]+q[96]*q[95]-q[52]*q[96]*q[95])*q[37]*q[32])+q[37]*q[33]-((q[52]+q[96]*q[95]-q[52]*q[96]*q[95])+q[37]*q[32]-(q[52]+q[96]*q[95]-q[52]*q[96]*q[95])*q[37]*q[32])*q[37]*q[33])+q[37]*q[34]-(((q[52]+q[96]*q[95]-q[52]*q[96]*q[95])+q[37]*q[32]-(q[52]+q[96]*q[95]-q[52]*q[96]*q[95])*q[37]*q[32])+q[37]*q[33]-((q[52]+q[96]*q[95]-q[52]*q[96]*q[95])+q[37]*q[32]-(q[52]+q[96]*q[95]-q[52]*q[96]*q[95])*q[37]*q[32])*q[37]*q[33])*q[37]*q[34])
            w[54]=q[53]*q[52]
            w[55]=q[53]*q[37]
            w[56]=(q[55]+q[98]-q[55]*q[98])
            w[57]=(q[9]+q[48]-q[9]*q[48])
            w[58]=q[9]*q[48]
            w[59]=q[9]*q[48]
            w[60]=(q[15]+q[56]-q[15]*q[56])
            w[61]=(q[72]*q[71]+q[69]-q[72]*q[71]*q[69])
            w[62]=q[61]
            w[63]=(q[61]*q[62]+q[51]-q[61]*q[62]*q[51])
            w[64]=q[63]
            w[65]=q[4]
            w[66]=((((q[65]+q[103]-q[65]*q[103])+q[73]-(q[65]+q[103]-q[65]*q[103])*q[73])+q[76]-((q[65]+q[103]-q[65]*q[103])+q[73]-(q[65]+q[103]-q[65]*q[103])*q[73])*q[76])+q[82]*q[83]-(((q[65]+q[103]-q[65]*q[103])+q[73]-(q[65]+q[103]-q[65]*q[103])*q[73])+q[76]-((q[65]+q[103]-q[65]*q[103])+q[73]-(q[65]+q[103]-q[65]*q[103])*q[73])*q[76])*q[82]*q[83])
            w[67]=q[66]
            w[68]=q[66]
            w[69]=q[68]
            w[70]=q[68]
            w[71]=q[69]
            w[72]=q[70]
            w[73]=q[78]
            w[74]=q[84]
            w[75]=q[84]
            w[76]=(q[75]+q[74]-q[75]*q[74])
            w[77]=(q[66]+q[35]-q[66]*q[35])
            w[78]=q[66]
            w[79]=q[77]
            w[80]=q[78]
            w[81]=p[11]
            w[82]=q[81]
            w[83]=q[81]
            w[84]=p[13]
            w[85]=(q[79]+q[9]-q[79]*q[9])
            w[86]=q[85]*(q[79]+q[9]-q[79]*q[9])*(1-q[89])
            w[87]=(q[85]*q[89]+q[85]*p[9]-q[85]*q[89]*q[85]*p[9])
            w[88]=((q[79]*q[92]+p[9]-q[79]*q[92]*p[9])+p[10]-(q[79]*q[92]+p[9]-q[79]*q[92]*p[9])*p[10])
            w[89]=(q[88]+(q[73]+q[79]-q[73]*q[79])*q[92]-q[88]*(q[73]+q[79]-q[73]*q[79])*q[92])*(1-q[86])
            w[90]=(q[86]+q[93]-q[86]*q[93])*p[16]*(1-q[92])
            w[91]=q[89]*p[15]
            w[92]=q[90]*(1-q[91])
            w[93]=(1-p[14])*q[79]
            w[94]=(p[17]+p[18]-p[17]*p[18])
            w[95]=q[94]
            w[96]=(q[95]+q[99]-q[95]*q[99])
            w[97]=q[50]*q[96]
            w[98]=q[53]*q[96]
            w[99]=(p[19]+q[101]-p[19]*q[101])
            w[100]=q[96]
            w[101]=q[9]
            w[102]=(q[102]+q[104]-q[102]*q[104])
            w[103]=q[9]
            w[104]=q[9]
            w[105]=q[9]
            w[106]=q[9]
            w[107]=q[9]
            dqdt= act_f(af, x=w, b=p[0], u = threshold_values)-decay_rates*q
            return dqdt
        return system_ode
    def solve(self, t_span=(0, 20), t_eval=None, grinitial_conditions=None, grmethod=None, grout_of_the_network_arguments_eval=None,af=None,decay_rates_list=None
              ,grthreshold_values=None):
        if grinitial_conditions is None:
            grinitial_conditions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.2, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if grmethod is None:
            grmethod = 'LSODA'
        if grout_of_the_network_arguments_eval is None:
            grout_of_the_network_arguments_eval = np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        if decay_rates_list is None:
            decay_rates_list = np.full(self.n,1)
        if af is None:
            af = 'sigmoid'
        if grthreshold_values is None:
            grthreshold_values = np.full(self.n,0.5)
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
            resultados = {
            'tiempos': sol.t,
            'genes': {}
            }
            for i, gen in enumerate(self.array_of_elements):
                resultados['genes'][gen] = {
                    'valores': sol.y[i],
                    'inicial': sol.y[i, 0],
                    'final': sol.y[i, -1],
                    'max': np.max(sol.y[i]),
                    'min': np.min(sol.y[i]),
                    'mean': np.mean(sol.y[i])
                }
        return resultados

class NetworkInterface(NetworkEquations):  # Asegúrate de que NetworkEquations esté definido
    def __init__(self, features: dict):
        super().__init__()
        self.value_sliders = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.2, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.max_sliders_out = [100] + 20 * [1]
        self.value_sliders_out = np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        self.distribution_type = None
        self.features = features
        self.subnetwork_options = list(features.keys())
        self.last_results = None
        self.gradio_interface()

    def plot_results(self, resultados: dict = None, options: list = None) -> go.Figure:
        if options is None:
            options = ['OXPHOS', 'IL10OUT', 'PLC', 'AMPK', 'Glycolysis']
        if len(options) > 16:
            fig = go.Figure()
            return go.Figure().update_layout(title=f"Error: Change one of the nodes to add it to your graphic")
        else:
            fig = go.Figure()
            t = resultados['tiempos']
            for option in options:
                fig.add_trace(go.Line(x=t, y=resultados['genes'][option]['valores'], mode='lines', name=option,
                                      line=dict(width=2),
                                      hovertemplate=f'{option}<br>Time: %{{x:.2f}}<br>Value: %{{y:.4f}}'))
            Titulo = {
                'text': "Nodes expression",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top'
            }
            fig.update_layout(
                title=Titulo,
                yaxis=dict(range=[-0.1, 1], autorange=False),
                xaxis_title='Time',
                yaxis_title='Expression',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
        return fig

    def design_and_personalize_subplots(self, resultados):
        n_co = 3
        n_rows = math.ceil(len(self.features.keys()) / n_co)
        # Cambio: hacer cada subplot cuadrado (3x3 pulgadas)
        fig2, axis = plt.subplots(n_rows, n_co, figsize=(5* n_co, 5.5* n_rows))
        axis = axis.flatten()
        categorias = list(self.features.keys())[:8]
        for idx, (axi, subr) in enumerate(zip(axis, categorias)):
            for inx, (gen, col, sty, label) in enumerate(zip(self.features[subr]['variables'],
                                                              self.features[subr]['colores'],
                                                              self.features[subr]['estilos'],
                                                              self.features[subr]['labels'])):
                data = resultados['genes'][gen]
                axi.plot(resultados['tiempos'], data['valores'], label=label,
                         color=col, linestyle=sty, linewidth=2)
                axi.legend(bbox_to_anchor=(0., 1.5, 1., .02), loc=1, ncol=1, mode="expand",
                           borderaxespad=0., prop={'size': 8}, title=subr, fontsize=8)
            axi.set_xlabel('Time', fontsize=8)
            axi.set_ylabel('Expression', fontsize=8)
            axi.set_ylim(-0.1, 1)
            axi.grid(True, alpha=0.3)
        for j in range(idx + 1, len(axis)):
            axis[j].set_visible(False)
        plt.tight_layout(h_pad=1.80)
        return fig2

    def create_Plot(self):
        if self.last_results is None:
            return go.Figure().update_layout(title='Run a network simulation first')
        else:
            Figura = self.design_and_personalize_subplots(self.last_results)
            return Figura

    def create_selected_subnetwork_plot(self, resultados: dict, subnetwork_name: str) -> go.Figure:
        if subnetwork_name not in self.features:
            return go.Figure().update_layout(title=f"Subnetwork '{subnetwork_name}' not found")
        config = self.features[subnetwork_name]
        fig = go.Figure()
        t = resultados['tiempos']
        for i, (var, label) in enumerate(zip(self.features[subnetwork_name]['variables'],
                                              self.features[subnetwork_name]['labels'])):
            if var in resultados['genes']:
                fig.add_trace(go.Line(
                    x=t,
                    y=resultados['genes'][var]['valores'],
                    mode='lines',
                    name=label,
                    line=dict(width=2),
                    hovertemplate=f'{label}<br>Time: %{{x:.2f}}<br>Value: %{{y:.4f}}'
                ))
        Titulo = {
            'text': f"{subnetwork_name}",
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        }
        fig.update_layout(
            title=Titulo,
            xaxis_title="Time",
            yaxis_title="Expression",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        return fig

    def gradio_interface(self, n_columns: int = 5):
        def update_title(a_function, method):
            return f"""
            <div style="background-color:#4c8aad; padding:10px; border-radius:5px;">
                <h3 style="margin:0; text-align: center;">Genetic Network Simulator</h3>
                <p style="margin:5px 0; text-align: center;">
                    Network: {self.n} genes |
                    Activation: {a_function} |
                    Method: {method}
                </p>
            </div>
            """

        def update_network_params_header(a_function, method):
            # Muestra información combinada de todos los parámetros de red
            return f"""
            <div style="background-color:#4c8aad; padding:10px; border-radius:5px;">
                <h3 style="margin:0; text-align: center;">Network Parameters</h3>
                <p style="margin:5px 0; text-align: center;">
                    Network: {self.n} genes |
                    Activation: {a_function} |
                    Method: {method} |
                    External inputs: {len(self.out_of_network_arguments)} |
                    Decay rates: {self.n} |
                    Thresholds: {self.n}
                </p>
            </div>
            """

        def update_interface(act, meth):
            return update_title(act, meth), update_network_params_header(act, meth)

        with gr.Blocks(title="Network simulator", theme='Taithrah/Minimal') as demo:
            with gr.Row(variant="compact"):
                gr.Markdown("Solving ODE rapidly")
                gr.Markdown("Exploring networks more efficiently")
            with gr.Row(variant="compact"):
                activation = gr.Dropdown(
                    choices=['sigmoid', 'tanh', 'relu'],
                    label="Activation function",
                    value='sigmoid'
                )
                method = gr.Dropdown(
                    choices=['TRBDF2', 'DOP853', 'RadauIIA5', 'Rodas5', 'LSODA'],
                    label="ODE Solver",
                    value='LSODA'
                )
            with gr.Row(variant="compact"):
                distribution = gr.Dropdown(
                    choices=['normal', 'uniform', 'laplace'],
                    value='uniform',
                    label='Distribution to simulate network dynamics'
                )
            with gr.Row(variant="compact"):
                solve_btn = gr.Button("Simulate Network", variant="primary")

            a_function = 'sigmoid'
            n_columns = 5

            # Definimos los sliders en el orden esperado por after_push_button:
            # 1. Initial conditions (todos_sliders_ic)
            # 2. out_of_network (all_sliders_out)
            # 3. decay rates (todos_sliders_alpha)
            # 4. thresholds (todos_sliders_umbral)
            # Luego activation, method y select_box se añaden al final en variables_to_control.
            todos_sliders_ic = []
            all_sliders_out = []
            todos_sliders_alpha = []
            todos_sliders_umbral = []

            with gr.Tabs():
                with gr.Tab("Initial conditions"):
                    title_html = gr.HTML(update_title(a_function, method))
                    n_filas_ic = (self.n + n_columns - 1) // n_columns
                    for j in range(n_filas_ic):
                        with gr.Row(variant="compact"):
                            inicio = j * 5
                            fin = min((j + 1) * 5, self.n)
                            for idx in range(inicio, fin):
                                gen = self.array_of_elements[idx]
                                val = self.value_sliders[idx]
                                slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    label=f'{gen}',
                                    value=val,
                                    interactive=True
                                )
                                todos_sliders_ic.append(slider)

                with gr.Tab("Network parameters"):
                    network_params_html = gr.HTML(update_network_params_header(a_function, method))
                    with gr.Row(variant="compact"):
                        with gr.Column():
                            gr.Markdown("### b parameter")
                        with gr.Column():
                            slider_b = gr.Slider(
                                        minimum=0.0,
                                        maximum=self.max_sliders_out[0],
                                        step=0.01,
                                        label=f'{self.out_of_network_arguments[0]}',
                                        value=self.value_sliders_out[0],
                                        interactive=True
                                    )
                        all_sliders_out.append(slider_b)
                    gr.Markdown("### Decay Rates")
                    n_filas_alpha = (self.n + n_columns - 1) // n_columns
                    for s in range(n_filas_alpha):
                        with gr.Row(variant="compact"):
                            start = s * n_columns
                            end = min((s + 1) * n_columns, self.n)
                            for idx in range(start, end):
                                par = self.array_of_elements[idx]
                                slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    label=f"Decay rate ({par})",
                                    value=1.0,  # valor por defecto
                                    interactive=True
                                )
                                todos_sliders_alpha.append(slider)

                    # --- Thresholds (antes en "Threshold values") ---
                    gr.Markdown("### Thresholds")
                    n_filas_threshold = (self.n + n_columns - 1) // n_columns
                    for s in range(n_filas_threshold):
                        with gr.Row(variant="compact"):
                            start = s * n_columns
                            end = min((s + 1) * n_columns, self.n)
                            for idx in range(start, end):
                                par = self.array_of_elements[idx]
                                slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    label=f"Threshold ({par})",
                                    value=0.5,
                                    interactive=True
                                )
                                todos_sliders_umbral.append(slider)
                with gr.Tab("Inputs"):

                    n_filas_out = (len(self.out_of_network_arguments[1:]) + n_columns - 1) // n_columns
                    for j in range(n_filas_out):
                        with gr.Row(variant="compact"):
                            inicio = j * 5
                            fin = min((j + 1) * 5, len(self.out_of_network_arguments[1:]))
                            for idx in range(inicio, fin):
                                out = self.out_of_network_arguments[1:][idx]
                                slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=self.max_sliders_out[1:][idx],
                                    step=0.01,
                                    label=f'{self.out_of_network_arguments[1:][idx]}',
                                    value=self.value_sliders_out[1:][idx],
                                    interactive=True
                                )
                                all_sliders_out.append(slider)
                # El resto de las pestañas se mantienen igual
                with gr.Tab("Plot customization"):
                    with gr.Row():
                        with gr.Column():
                            subnetwork_selector = gr.Dropdown(
                                choices=self.subnetwork_options,
                                value=self.subnetwork_options[0] if self.subnetwork_options else "",
                                label="Choose Subnetwork to display",
                                interactive=True
                            )
                            microplots = gr.Plot()
                    with gr.Row():
                        with gr.Column():
                            select_box = gr.CheckboxGroup(
                                choices=list(self.array_of_elements),
                                value=['OXPHOS', 'IL10OUT', 'PLC', 'AMPK', 'Glycolysis'],
                                label="Choose the nodes to plot (15 nodes max)"
                            )
                        with gr.Column():
                            figura = gr.Plot()
                            select_box.change(
                                fn=self.update_general_plot,
                                inputs=select_box,
                                outputs=figura
                            )
                with gr.Tab("For papers"):
                    gr.Markdown("Summary of the results for papers and other publications")
                    with gr.Column():
                        GButton = gr.Button("Get plots for papers", variant="primary")
                        GPlot = gr.Plot()

                with gr.Tab("General information"):
                    gr.Markdown("Information about the network proposal")



            # Lista completa de entradas para el botón Solve
            variables_to_control = (todos_sliders_ic +
                                    all_sliders_out +
                                    todos_sliders_alpha +
                                    todos_sliders_umbral +
                                    [activation] +
                                    [method] +
                                    [select_box])

            # Eventos de cambio
            activation.change(
                fn=update_interface,
                inputs=[activation, method],
                outputs=[title_html, network_params_html]
            )
            method.change(
                fn=update_interface,
                inputs=[activation, method],
                outputs=[title_html, network_params_html]
            )
            distribution.change(
                fn=self.set_distribution,
                inputs=distribution,
                outputs=todos_sliders_ic
            )
            solve_btn.click(
                fn=self.after_push_button,
                inputs=variables_to_control,
                outputs=[figura, microplots]
            )
            subnetwork_selector.change(
                fn=self.update_selected_subnetwork,
                inputs=subnetwork_selector,
                outputs=microplots
            )
            GButton.click(
                fn=self.create_Plot,
                outputs=GPlot
            )

        return demo.launch()

    # Las siguientes funciones no requieren cambios
    def update_general_plot(self, selected_nodes: list) -> go.Figure:
        if not hasattr(self, 'last_results') or self.last_results is None:
            return go.Figure().update_layout(title="Please run a simulation first to plot selected nodes.")
        return self.plot_results(self.last_results, selected_nodes)

    def after_push_button(self, *values):
        genes = values[:self.n]
        out_params = values[self.n:self.n + len(self.out_of_network_arguments)]
        decay_rates = values[self.n + len(self.out_of_network_arguments):self.n + len(self.out_of_network_arguments) + self.n]
        threshold_values = values[self.n + len(self.out_of_network_arguments) + self.n:self.n + len(self.out_of_network_arguments) + self.n + self.n]
        activation = values[self.n + self.n + self.n + len(self.out_of_network_arguments)]
        metodo = values[self.n + self.n + self.n + len(self.out_of_network_arguments) + 1]
        selection_box_plot = values[self.n + self.n + self.n + len(self.out_of_network_arguments) + 2]

        Resultados = self.solve(
            t_span=(0, 20),
            t_eval=None,
            grinitial_conditions=genes,
            grmethod=metodo,
            grout_of_the_network_arguments_eval=out_params,
            af=activation,
            decay_rates_list=decay_rates,
            grthreshold_values=threshold_values
        )
        self.last_results = Resultados

        Fig1 = self.plot_results(Resultados, selection_box_plot)
        Fig2 = self.update_selected_subnetwork(self.subnetwork_name)
        return Fig1, Fig2

    def update_selected_subnetwork(self, subnetwork_name=None) -> go.Figure:
        if subnetwork_name is None:
            subnetwork_name = 'Metabolism'
        if subnetwork_name not in self.features:
            return go.Figure().update_layout(title=f"Subnetwork '{subnetwork_name}' not found")
        Fig2 = self.create_selected_subnetwork_plot(self.last_results, subnetwork_name)
        self.subnetwork_name = subnetwork_name
        return Fig2

    def set_distribution(self, option):
        if option == 'normal':
            return np.absolute(np.random.normal(0.5, 0.15, self.n)).tolist()
        elif option == 'uniform':
            return np.absolute(np.random.uniform(0.5, 0.15, self.n)).tolist()
        elif option == 'laplace':
            return np.absolute(np.random.laplace(0.5, 0.15, self.n)).tolist()

if __name__ == "__main__":
    Subredes = {}

     # 1. METABOLISMO
    metabolismo_vars = ['Glycolysis', 'OXPHOS', 'AMPATPratio', 'AMPK']
    metabolismo_labels = ['Glycolysis', 'OXPHOS', 'AMP/ATP Ratio', 'AMPK']
    metabolismo_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    metabolismo_estilos = ['-', ':', '--', '-.']
    Subredes['Metabolism'] = {'variables': metabolismo_vars, 'labels': metabolismo_labels,
                              'colores': metabolismo_colores, 'estilos': metabolismo_estilos}

    # 2. ACTIVIDAD DE MACRÓFAGOS
    macroactividad_vars = ['PHAGOCYTOSIS', 'PROCESSING', 'MHC2', 'TLR4END']
    macroactividad_labels = ['Phagocytosis', 'Antigen Processing', 'MHC II', 'Endocytosis']
    macroactividad_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    macroactividad_estilos = ['-', ':', '--', '-.']
    Subredes['Macroactivity'] = {'variables': macroactividad_vars, 'labels': macroactividad_labels,
                                 'colores': macroactividad_colores, 'estilos': macroactividad_estilos}

    # 3. FACTORES DE TRANSCRIPCIÓN
    transcripcion_vars = ['NFKB', 'AP1', 'IRF3', 'IRF4', 'IRF7']
    transcripcion_labels = ['NF-κB', 'AP-1', 'IRF3', 'IRF4', 'IRF7']
    transcripcion_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red', 'black']
    transcripcion_estilos = ['-', ':', '--', '-.', '-']
    Subredes['Transcription Factors'] = {'variables': transcripcion_vars, 'labels': transcripcion_labels,
                                          'colores': transcripcion_colores, 'estilos': transcripcion_estilos}

    # 4. MACRÓFAGOS M1
    m1_vars = ['IFNABOUT', 'IL1BOUT', 'IL6OUT', 'IL12OUT', 'TNFAOUT']
    m1_labels = ['IFN-α', 'IL-1β', 'IL-6', 'IL-12', 'TNF-α']
    m1_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red', 'black']
    m1_estilos = ['-', ':', '--', '-.', '-']
    Subredes['M1 macrophages'] = {'variables': m1_vars, 'labels': m1_labels,
                                'colores': m1_colores, 'estilos': m1_estilos}

    # 5. MACRÓFAGOS M2a
    m2a_vars = ['PPARG', 'IL10OUT', 'JMJD3', 'STAT6']
    m2a_labels = ['PPAR-γ', 'IL-10', 'JMJD3', 'STAT6']
    m2a_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    m2a_estilos = ['-', ':', '--', '-.']
    Subredes['M2a macrophages'] = {'variables': m2a_vars, 'labels': m2a_labels,
                                 'colores': m2a_colores, 'estilos': m2a_estilos}

    # 6. MACRÓFAGOS M2b
    m2b_vars = ['ERK', 'IL10OUT']
    m2b_labels = ['ERK', 'IL-10']
    m2b_colores = ['lawngreen', 'blueviolet']
    m2b_estilos = ['-', ':']
    Subredes['M2b Macrophages'] = {'variables': m2b_vars, 'labels': m2b_labels,
                                 'colores': m2b_colores, 'estilos': m2b_estilos}

    # 7. MACRÓFAGOS M2c
    m2c_vars = ['STAT3', 'IL10OUT']
    m2c_labels = ['STAT3', 'IL-10']
    m2c_colores = ['lawngreen', 'blueviolet']
    m2c_estilos = ['-', ':']
    Subredes['M2c macrophages'] = {'variables': m2c_vars, 'labels': m2c_labels,
                                 'colores': m2c_colores, 'estilos': m2c_estilos}

    # 8. CREB Y GSK3B
    creb_vars = ['CREB1', 'GSK3B', 'AKT1', 'AKT3']
    creb_labels = ['CREB', 'GSK3B', 'AKT1', 'AKT3']
    creb_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    creb_estilos = ['-', ':', '--', '-.']
    Subredes['Creb Gsk3b'] = {'variables': creb_vars, 'labels': creb_labels,
                             'colores': creb_colores, 'estilos': creb_estilos}
    network = NetworkInterface(Subredes)
