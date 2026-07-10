import numpy as np
import math
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
"""From this file weights_parsed.txt we get the function system ode to run the code """
"""If you want to avoid any inconvenience with file_paths you can use this version. Due to the amount of different errors you might encounter I recommend using this version
and input the arrays manually"""
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
        self.array_of_elements = np.array(['TCR', 'CD28', 'AP1', 'CD25', 'IL2G', 'IL2E', 'MTOR',
                                           'ZAP70', 'STAT5', 'NFAT', 'NFKB', 'AKT', 'CTLA4',
                                           'CTLA4DIM', 'BCL2', 'NDRG1', 'DAG', 'SOS', 'RASGTPR',
                                           'LCK', 'PDK1', 'LAT', 'PLC', 'PI3K', 'PIP2', 'PIP3',
                                           'IP3', 'CA', 'PKC', 'TBET', 'IFNG', 'GATA3', 'IL4',
                                           'FOXP3', 'IL10', 'TGFB', 'RORGT', 'IL21', 'IL17',
                                           'BCL6', 'IL9', 'CD40L', 'MTORC1', 'MTORC2', 'LKB1',
                                           'AMPK', 'Glycolysis', 'GLUTAMINOLISIS', 'AKG', 'OXPHOS',
                                           'AMPATPratio', 'HIF1A'])
        self.n = len(self.array_of_elements)
        return self.array_of_elements
    def networkparameters(self):
        """I think it is worth mentioning this method has to be optional
        for many networks without external parameters.However, we can adjust it later"""
        self.out_of_network_arguments = np.array(['b', 'AttAnt', 'TAnt', 'AttCD8086', 'TCD8086', 'IFNGE', 'IL12E', 'IL18E',
                                                  'IL33E', 'IL4E', 'TGFBE', 'IL10E', 'IL21E',
                                                  'IL6E', 'GLC', 'GLN', 'FA', 'TRP', 'O2', 'METF',
                                                  'RAPA', 'PRED','CS'])

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
            w[0]=(p[1]/(1+np.exp(t-p[2])))* (1 - q[13])
            w[1]=(p[3]/(1+np.exp(t-p[4]))) * (1 - q[13])
            w[2]=q[18] * (1 - p[21])
            w[3]=q[4] * (1 - q[13])
            w[4]=q[9] * q[2] * (1 - q[15])
            w[5]=q[4]
            w[6]=(q[3] + q[11]) - (q[3] * q[11])
            w[7]=(q[0] * q[19]) * (1 - q[13]) * (1 - p[21])
            w[8]=q[3] * (1 - q[13])
            w[9]=(q[27]) * (1 - p[22])
            w[10]=q[28] * (1 - p[21])
            w[11]=(((q[1]) * (1 - q[13])) + (q[20])) - (((q[1]) * (1 - q[13])) * (q[20]))
            w[12]=q[4] * q[7]
            w[13]=((q[12] * (p[3]/(1+np.exp(t-p[4])))) + (q[33] * q[35])) - ((q[12] * (p[3]/(1+np.exp(t-p[4])))) * (q[33] * q[35]))
            w[14]=q[11]
            w[15]=q[9] * (1 - q[11])
            w[16]=q[22] * q[24]
            w[17]=q[1]
            w[18]=((q[21] * q[17] * q[16]) + (q[3] * q[16])) - ((q[21] * q[17] * q[16]) * (q[3] * q[16]))
            w[19]=q[0] * (1 - q[13])
            w[20]=(q[3] + q[1] + q[25]) - (q[3] * q[1]) - (q[1] * q[25]) - (q[3] * q[25]) + (q[3] * q[1] * q[25])
            w[21]=q[7]
            w[22]=(q[7] + q[3]) - (q[7] * q[3])
            w[23]=(q[7] + q[3]) - (q[7] * q[3])
            w[24]=(q[23] + q[22]) - (q[23] * q[22])
            w[25]=q[24]
            w[26]=q[24] * q[22]
            w[27]=q[26]
            w[28]=q[16]
            w[29]=(p[8] * p[7] * p[6] * q[5] * p[5] * q[42] * q[10] * q[9] * q[2] * p[17] * q[48]) * (1 - q[32]) * (1 - q[34]) * (1 - q[31])
            w[30]=(q[29] * q[2] * q[9]) * (1 - q[31])
            w[31]=(p[8] * p[9] * q[5] * q[43] * q[8] * q[9]) * (1 - q[29]) * (1 - q[35]) * (1 - q[30]) * (1 - q[39])
            w[32]=(q[31]) * (1 - q[29]) * (1 - q[30])
            w[33]=(((p[10] * p[11] * q[9] * q[8] * q[2] * q[5]) + (p[10] * p[11] * q[34] * q[12]) + (p[10] * q[35])) - ((p[10] * p[11] * q[9] * q[8] * q[2] * q[5]) * (p[10] * p[11] * q[34] * q[12])) - ((p[10] * p[11] * q[34] * q[12]) * (p[10] * q[35])) - ((p[10] * p[11] * q[9] * q[8] * q[2] * q[5]) * (p[10] * q[35])) + ((p[10] * p[11] * q[9] * q[8] * q[2] * q[5]) * (p[10] * p[11] * q[34] * q[12]) * (p[10] * q[35]))) * (1 - q[30]) * (1 - q[51]) * (1 - p[13])
            w[34]=p[10] * q[33]
            w[35]=q[33]
            w[36]=(((p[13] * p[12] * p[10] * q[2] * q[42] * p[17]) + (p[12] * p[10] * q[2] * q[42] * q[48]) + (q[51])) - ((p[13] * p[12] * p[10] * q[2] * q[42] * p[17]) * (p[12] * p[10] * q[2] * q[42] * q[48])) - ((p[12] * p[10] * q[2] * q[42] * q[48]) * (q[51])) - ((p[13] * p[12] * p[10] * q[2] * q[42] * p[17]) * (q[51])) + ((p[13] * p[12] * p[10] * q[2] * q[42] * p[17]) * (p[12] * p[10] * q[2] * q[42] * q[48]) * (q[51]))) * (1 - q[29]) * (1 - q[33]) * (1 - q[31])
            w[37]=(((p[12] * q[36]) + (p[13] * q[39])) - ((p[12] * q[36]) * (p[13] * q[39]))) * (1 - q[30]) * (1 - q[32]) * (1 - q[34])
            w[38]=q[36]
            w[39]=(p[13] * p[12] * q[2] * q[42]) * (1 - q[36]) * (1 - q[29]) * (1 - q[31])
            w[40]=q[39]
            w[41]=q[39]
            w[42]=(((q[6] * q[11]) + (q[6] * q[48])) - ((q[6] * q[11]) * (q[6] * q[48]))) * (1 - q[45]) * (1 - p[20])
            w[43]=((q[6] * q[45]) + (q[6] * p[9])) - ((q[6] * q[45]) * (q[6] * p[9]))
            w[44]=(q[11] * q[50])
            w[45]=(q[44] * (1 - q[42]) + q[27] * q[50] * (1 - q[42]) + q[11] * q[50] * (1 - q[42]) + q[33] + q[39] + p[19]) - ((q[44] * (1 - q[42]) * q[27] * q[50] * (1 - q[42])) - (q[44] * (1 - q[42]) * q[11] * q[50] * (1 - q[42])) - (q[44] * (1 - q[42]) * q[33]) - (q[44] * (1 - q[42]) * q[39]) - (q[44] * (1 - q[42]) * p[19]) - (q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42])) - (q[27] * q[50] * (1 - q[42]) * q[33]) - (q[27] * q[50] * (1 - q[42]) * q[39]) - (q[27] * q[50] * (1 - q[42]) * p[19]) - (q[11] * q[50] * (1 - q[42]) * q[33]) - (q[11] * q[50] * (1 - q[42]) * q[39]) - (q[11] * q[50] * (1 - q[42]) * p[19]) - (q[33] * q[39]) - (q[33] * p[19]) - (q[39] * p[19])) + ((q[44] * (1 - q[42]) * q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42])) + (q[44] * (1 - q[42]) * q[27] * q[50] * (1 - q[42]) * q[33]) + (q[44] * (1 - q[42]) * q[27] * q[50] * (1 - q[42]) * q[39]) + (q[44] * (1 - q[42]) * q[27] * q[50] * (1 - q[42]) * p[19]) + (q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[33]) + (q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[39]) + (q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * p[19]) + (q[11] * q[50] * (1 - q[42]) * q[33] * q[39]) + (q[11] * q[50] * (1 - q[42]) * q[33] * p[19]) + (q[33] * q[39] * p[19])) - ((q[44] * (1 - q[42]) * q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[33]) - (q[44] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[33] * q[39]) - (q[44] * (1 - q[42]) * q[33] * q[39] * p[19]) - (q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[33] * q[39]) - (q[27] * q[50] * (1 - q[42]) * q[33] * q[39] * p[19]) - (q[11] * q[50] * (1 - q[42]) * q[33] * q[39] * p[19])) + ((q[44] * (1 - q[42]) * q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[33] * q[39]) + (q[44] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[33] * q[39] * p[19]) + (q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[33] * q[39] * p[19])) - (q[44] * (1 - q[42]) * q[27] * q[50] * (1 - q[42]) * q[11] * q[50] * (1 - q[42]) * q[33] * q[39] * p[19])
            w[46]=((((q[42] * p[14]) + (q[51] * p[14])) - ((q[42] * p[14]) * (q[51] * p[14]))) * (1 -q[50]) * (1 - q[39]))
            w[47]=p[15]
            w[48]=q[47]
            w[49]=q[45] * p[16]
            w[50]=q[46] * (1 - q[49])
            w[51]=(1-p[18]) * q[11]
            dqdt= act_f(af, x=w, b=p[0], u = threshold_values)-decay_rates*q
            return dqdt
        return system_ode
    def solve(self, t_span=(0, 30), t_eval=None, grinitial_conditions=None, grmethod=None, grout_of_the_network_arguments_eval=None,af=None,decay_rates_list=None
              ,grthreshold_values=None):
        if grinitial_conditions is None:
            grinitial_conditions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.2, 0, 0, 0, 0])
            grmethod = 'LSODA'
        if grout_of_the_network_arguments_eval is None:
            grout_of_the_network_arguments_eval = np.array([10,1,15,1,15,15,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])
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
import matplotlib.pyplot as plt
import math
import gradio as gr
import random
class NetworkInterface(NetworkEquations):
    def __init__(self,features :dict):
        super().__init__()
        self.value_sliders = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.2, 0, 0, 0, 0])
        self.max_sliders_out = [100]+[1,100,1,100]+18*[1]
        self.value_sliders_out = np.array([10,1,15,1,15,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0])
        self.distribution_type = None
        self.features = features
        self.subnetwork_options = list(features.keys())
        self.last_results = None # Initialize last_results
        self.gradio_interface()

    """ mathplotlib was changed to pyplot by doing this we are improving the performance, instead of generating
    the 165 plot we just store the data and plot it as the user requests it"""
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

    def design_and_personalize_subplots(self, resultados,data_to_plot_ext):
        n_co = 3
        n_rows = math.ceil(len(self.features.keys()) / n_co)
        # Cambio: hacer cada subplot cuadrado (3x3 pulgadas)
        fig2, axis = plt.subplots(n_rows, n_co, figsize=(5.5* n_co, 6.4* n_rows))
        axis = axis.flatten()
        self.reference_external=dict(zip(self.out_of_network_arguments,data_to_plot_ext))
        axis[0].set_xticks(range(0, len(list(self.reference_external.keys())[1:])))
        axis[0].set_xticklabels(list(self.reference_external.keys())[1:], rotation=90, ha='right', fontsize=8)
        axis[0].set_ylim(0,1)
        axis[0].bar(list(self.reference_external.keys())[1:], data_to_plot_ext[1:], color='purple')
        axis[0].legend(bbox_to_anchor=(0., 1.3, 1., .09), loc=1, ncol=1, mode="expand",
                           borderaxespad=0., prop={'size': 8}, title='Input expression level', fontsize=8)
        categorias = list(self.features.keys())[:8]
        for idx, (axi, subr) in enumerate(zip(axis[1:], categorias)):
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
        plt.tight_layout(h_pad=1.80)
        return fig2

    def create_Plot(self,*ext_parameters):
        if self.last_results is None:
            return go.Figure().update_layout(title='Run a network simulation first')
        else:
            Figura = self.design_and_personalize_subplots(self.last_results,ext_parameters)
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
                                value=random.choices(self.array_of_elements,k=6),
                                label="Choose the nodes to plot (15 nodes max)"
                            )
                        with gr.Column():
                            figura = gr.Plot()
                            select_box.change(
                                fn=self.update_general_plot,
                                inputs=select_box,
                                outputs=figura
                            )
                with gr.Tab("Results"):
                    gr.Markdown("Summary of the results for papers and other publications")
                    with gr.Column():
                        GButton = gr.Button("Plot Th phenotypes", variant="primary")
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
                inputs=all_sliders_out,
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
      # 12. FACTORES DE TRANSCRIPCIÓN DE CÉLULAS T
    t_transcripcion_vars = ['AP1', 'NFAT', 'NFKB']
    t_transcripcion_labels = ['AP1', 'NFAT', 'NFKB']
    t_transcripcion_colores = ['lawngreen', 'blueviolet', 'tab:cyan']
    t_transcripcion_estilos = ['-', ':', '--']
    Subredes['T cells FT'] = {'variables': t_transcripcion_vars, 'labels': t_transcripcion_labels,
                                  'colores': t_transcripcion_colores, 'estilos': t_transcripcion_estilos}

    # 13. MARCADORES DE ACTIVACIÓN
    activacion_vars = ['IL2G', 'MTORC1', 'MTORC2']
    activacion_labels = ['IL2G', 'MTORC1', 'MTORC2']
    activacion_colores = ['lawngreen', 'blueviolet', 'tab:cyan']
    activacion_estilos = ['-', ':', '--']
    Subredes['Activation markers'] = {'variables': activacion_vars, 'labels': activacion_labels,
                                        'colores': activacion_colores, 'estilos': activacion_estilos}

    # 14. CÉLULAS Th1
    th1_vars = ['TBET', 'IFNG']
    th1_labels = ['TBET', 'IFNG']
    th1_colores = ['lawngreen', 'blueviolet']
    th1_estilos = ['-', ':']
    Subredes['Th1'] = {'variables': th1_vars, 'labels': th1_labels,
                      'colores': th1_colores, 'estilos': th1_estilos}

    # 15. CÉLULAS Th2
    th2_vars = ['GATA3', 'IL4']
    th2_labels = ['GATA3', 'IL4']
    th2_colores = ['lawngreen', 'blueviolet']
    th2_estilos = ['-', ':']
    Subredes['Th2'] = {'variables': th2_vars, 'labels': th2_labels,
                      'colores': th2_colores, 'estilos': th2_estilos}

    # 16. CÉLULAS Th17
    th17_vars = ['IL17', 'IL21', 'RORGT']
    th17_labels = ['IL17', 'II21', 'RORGT']
    th17_colores = ['lawngreen', 'blueviolet', 'tab:cyan']
    th17_estilos = ['-', ':', '--']
    Subredes['Th17'] = {'variables': th17_vars, 'labels': th17_labels,
                       'colores': th17_colores, 'estilos': th17_estilos}

    # 17. CÉLULAS Treg
    treg_vars = ['FOXP3', 'TGFB', 'CTLA4DIM']
    treg_labels = ['FOXP3', 'TGFB', 'CTLA4DIM']
    treg_colores = ['lawngreen', 'blueviolet', 'tab:cyan']
    treg_estilos = ['-', ':', '--']
    Subredes['Treg'] = {'variables': treg_vars, 'labels': treg_labels,
                       'colores': treg_colores, 'estilos': treg_estilos}

    # 18. CÉLULAS Tfh
    tfh_vars = ['BCL6', 'IL21', 'CD40L', 'IL9']
    tfh_labels = ['BCL6', 'IL21', 'CD40L', 'IL9']
    tfh_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    tfh_estilos = ['-', ':', '--', '-.']
    Subredes['Tfh'] = {'variables': tfh_vars, 'labels': tfh_labels,
                      'colores': tfh_colores, 'estilos': tfh_estilos}

    network = NetworkInterface(Subredes)
