import onnx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class QuantizationErrorAnalyzer:
    
    # 1) ONNX file path
    PATHS = {
        'fp16'     : '../onnx_graph/facebook__opt-1.3b/fp16_baseline/model.onnx',
        # --------------------------------------per-channel--------------------------------------------- 
        'per-channel':{
        # -- bit-width = 8 --
        'int8_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int8_asym/w_8_gs_none/model.onnx', 
        'int8'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int8/w_8_gs_none/model.onnx',
        'fp8_e3m4' : '../onnx_graph/facebook__opt-1.3b/per-channel/fp8_e3m4/w_8_gs_none/model.onnx', 
        'fp8_e4m3' : '../onnx_graph/facebook__opt-1.3b/per-channel/fp8_e4m3/w_8_gs_none/model.onnx', 
        # -- bit-width = 7 --
        'int7_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int7_asym/w_7_gs_none/model.onnx',
        'int7'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int7/w_7_gs_none/model.onnx',
        # -- bit-width = 6 --
        'int6_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int6_asym/w_6_gs_none/model.onnx',
        'int6'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int6/w_6_gs_none/model.onnx',
        'fp6e3m2'  : '../onnx_graph/facebook__opt-1.3b/per-channel/fp6e3m2/w_6_gs_none/model.onnx',
        'fp6e2m3'  : '../onnx_graph/facebook__opt-1.3b/per-channel/fp6e2m3/w_6_gs_none/model.onnx',
        # -- bit-width = 5 --
        'int5_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int5_asym/w_5_gs_none/model.onnx',
        'int5'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int5/w_5_gs_none/model.onnx',
        'fp5e2m2'  : '../onnx_graph/facebook__opt-1.3b/per-channel/fp5e2m2/w_5_gs_none/model.onnx',
        'fp5e3m1'  : '../onnx_graph/facebook__opt-1.3b/per-channel/fp5e3m1/w_5_gs_none/model.onnx',
        # -- bit-width = 4 --
        'int4_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int4_asym/w_4_gs_none/model.onnx',
        'int4'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int4/w_4_gs_none/model.onnx',
        'fp4'      : '../onnx_graph/facebook__opt-1.3b/per-channel/fp4/w_4_gs_none/model.onnx',
        # -- bit-width = 3 --
        'int3_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int3_asym/w_3_gs_none/model.onnx',
        'int3'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int3/w_3_gs_none/model.onnx',
        'fp3'      : '../onnx_graph/facebook__opt-1.3b/per-channel/fp3/w_3_gs_none/model.onnx',
        },
        
        'per-group':{
        #---------------------------------------per-group--------------------------------------------
        # -- bit-width = 8 --
        'int8_asym': '../onnx_graph/facebook__opt-1.3b/per-group/int8_asym/w_8_gs_128/model.onnx', 
        'int8'     : '../onnx_graph/facebook__opt-1.3b/per-group/int8/w_8_gs_128/model.onnx',
        'fp8_e3m4' : '../onnx_graph/facebook__opt-1.3b/per-group/fp8_e3m4/w_8_gs_128/model.onnx', 
        'fp8_e4m3' : '../onnx_graph/facebook__opt-1.3b/per-group/fp8_e4m3/w_8_gs_128/model.onnx', 
        # -- bit-width = 7 --
        'int7_asym': '../onnx_graph/facebook__opt-1.3b/per-group/int7_asym/w_7_gs_128/model.onnx',
        'int7'     : '../onnx_graph/facebook__opt-1.3b/per-group/int7/w_7_gs_128/model.onnx',
        # -- bit-width = 6 --
        'int6_asym': '../onnx_graph/facebook__opt-1.3b/per-group/int6_asym/w_6_gs_128/model.onnx',
        'int6'     : '../onnx_graph/facebook__opt-1.3b/per-group/int6/w_6_gs_128/model.onnx',
        'fp6e3m2'  : '../onnx_graph/facebook__opt-1.3b/per-group/fp6e3m2/w_6_gs_128/model.onnx',
        'fp6e2m3'  : '../onnx_graph/facebook__opt-1.3b/per-group/fp6e2m3/w_6_gs_128/model.onnx',
        # -- bit-width = 5 --
        'int5_asym': '../onnx_graph/facebook__opt-1.3b/per-group/int5_asym/w_5_gs_128/model.onnx',
        'int5'     : '../onnx_graph/facebook__opt-1.3b/per-group/int5/w_5_gs_128/model.onnx',
        'fp5e2m2'  : '../onnx_graph/facebook__opt-1.3b/per-group/fp5e2m2/w_5_gs_128/model.onnx',
        'fp5e3m1'  : '../onnx_graph/facebook__opt-1.3b/per-group/fp5e3m1/w_5_gs_128/model.onnx',
        # -- bit-width = 4 --
        'int4_asym': '../onnx_graph/facebook__opt-1.3b/per-group/int4_asym/w_4_gs_128/model.onnx',
        'int4'     : '../onnx_graph/facebook__opt-1.3b/per-group/int4/w_4_gs_128/model.onnx',
        'fp4'      : '../onnx_graph/facebook__opt-1.3b/per-group/fp4/w_4_gs_128/model.onnx',
        # -- bit-width = 3 --
        'int3_asym': '../onnx_graph/facebook__opt-1.3b/per-group/int3_asym/w_3_gs_128/model.onnx',
        'int3'     : '../onnx_graph/facebook__opt-1.3b/per-group/int3/w_3_gs_128/model.onnx',
        'fp3'      : '../onnx_graph/facebook__opt-1.3b/per-group/fp3/w_3_gs_128/model.onnx',
        }

    }

    def __init__(self, quant_scheme: str, quant_key: str, tensor_name: str, analysis_range: tuple = None):
        if quant_scheme not in self.PATHS:
            raise ValueError(f"'{quant_scheme}' is not a valid granularity. choose only one of two. ()'per-channel', 'per-group')")
        if quant_key not in self.PATHS[quant_scheme]:
            raise ValueError(f"No '{quant_key}' datatype in granularity '{quant_scheme}'")
        self.quant_scheme = quant_scheme
        self.quant_key = quant_key
        self.tensor_name = tensor_name
        self.analysis_range = analysis_range
        
        quant_path = self.PATHS[quant_scheme][quant_key]
        
        # FP16 and selected datatype load
        self.w_fp16 = self._load_weight(self.PATHS['fp16'], self.tensor_name)
        self.w_quant = self._load_weight(quant_path, self.tensor_name)
        
        # error  
        self.elementwise_error = (self.w_fp16 - self.w_quant).flatten()
        self.abs_elementwise_error = np.abs(self.elementwise_error)
        self.indices = np.arange(self.elementwise_error.size, dtype=int)
        
        # error slicing according to configured weight index range
        self.sliced_elementwise_error = None
        self.sliced_abs_error = None
        if self.analysis_range:
            start, end = self.analysis_range
            if end > self.elementwise_error.size:
                print(f"[Warning] Max of analyzing range exceeds the tensor size. The Max range will be truncated as tensor max size")
                end = self.elementwise_error.size
            self.sliced_elementwise_error = self.elementwise_error[start:end]
            self.sliced_abs_error = self.abs_elementwise_error[start:end]
            
    def _load_weight(self, onnx_path: str, tensor_name: str) -> np.ndarray:
        print(f"Load '{tensor_name}' from path '{onnx_path}'")
        model = onnx.load(onnx_path)
        for init in model.graph.initializer:
            if init.name == tensor_name:
                return onnx.numpy_helper.to_array(init).astype(np.float32)
        raise KeyError(f"cannot find '{tensor_name}' in path"'{onnx_path}'"")

    def get_p99_error(self) -> float:
        return np.percentile(self.abs_elementwise_error, 99)

    def get_error_sum(self) -> float:
        return np.sum(self.abs_elementwise_error)

    def get_sliced_mse(self) -> float:
        if self.sliced_elementwise_error is not None:
            return np.mean(np.square(self.sliced_elementwise_error))
        return 0.0
    
    def get_sliced_abs_error_sum(self) -> float:
        if self.sliced_abs_error is not None:
            return np.sum(self.sliced_abs_error)
        return 0.0
    
    def print_statistics(self):
        print("\n--- result ---")
        print(f"Analyzing scheme: '{self.quant_scheme}'")
        print(f"Analyzing datatype: '{self.quant_key}'")
        print(f"Tensor: '{self.tensor_name}'")
        print(f"Sum of Absolute Error: {self.get_error_sum():.6f}")
        print(f"P99 of Absolute Error: {self.get_p99_error():.6f}")
        print("---------------------------------\n")
        
        if self.analysis_range:
            sliced_mse = self.get_sliced_mse()
            sliced_sum = self.get_sliced_abs_error_sum()
            start, end = self.analysis_range

            print(f"--- customized range [ {start} ~ {end-1} ] result ---")
            print(f"MSE: {sliced_mse:.15f}")
            print(f"Abs Error Sum: {sliced_sum:.6f}")
            print("---------------------------------\n")
       
    def generate_plots(self, plot_config: dict, plot_type: str = 'all'):
    
        print(f"'{plot_type}' graph is creating...")

        # 1. Element-wise Error 
        if plot_type in ['elementwise', 'all']:
            self._plot_line(
                data=self.elementwise_error,
                config=plot_config.get('elementwise_error_plot', {})
            )
    
        # 2. Absolute Element-wise Error 
        if plot_type in ['absolute', 'all']:
            self._plot_line(
                data=self.abs_elementwise_error,
                config=plot_config.get('abs_error_plot', {})
            )
    
        # 3. Absolute Error histogram
        if plot_type in ['histogram', 'all']:
            self._plot_histogram(
                data=self.abs_elementwise_error,
                config=plot_config.get('histogram_plot', {})
            )

        if plot_type not in ['elementwise', 'absolute', 'histogram', 'all']:
            print(f"'{plot_type}'is not valid type. Choose one of those. 'elementwise', 'absolute', 'histogram', 'all'")
            return 

        plt.show()
        print("Graph creation is completed.")
        
    def _plot_line(self, data: np.ndarray, config: dict):
        fig, ax = plt.subplots(figsize=config.get('figsize', (12, 6)))
        
        ax.plot(self.indices, data, 
                lw=config.get('linewidth', 2.0), 
                color=config.get('color', 'blue'),
                label=f"{self.quant_key}")

        ax.set_title(config.get('title', 'Plot'), fontsize=16)
        ax.set_xlabel(config.get('xlabel', 'Element ID'), fontsize=12)
        ax.set_ylabel(config.get('ylabel', 'Error'), fontsize=12)
            
        ax.set_xticks(np.arange(0, 100, 10))
        ax.xaxis.set_major_locator(MultipleLocator(10))
            
        ax.set_yticks(np.arange(0, 0.0005, 0.000125))
        ax.yaxis.set_major_locator(MultipleLocator(0.000125))
        
        if 'xlim' in config: ax.set_xlim(config['xlim'])
        if 'ylim' in config: ax.set_ylim(config['ylim'])
            
        if 'xticks_major' in config: ax.xaxis.set_major_locator(MultipleLocator(config['xticks_major']))
        if 'yticks_major' in config: ax.yaxis.set_major_locator(MultipleLocator(config['yticks_major']))
        
        ax.grid(which='major', linestyle='--', linewidth=0.5)
        ax.legend(loc='upper right')
        plt.tight_layout()

    def _plot_histogram(self, data: np.ndarray, config: dict):
        plt.figure(figsize=config.get('figsize', (10, 6)))
        
        plt.hist(data, 
                 bins=config.get('bins', 100), 
                 color=config.get('color', 'green'),
                 alpha=0.75,
                 label=f"{self.quant_key} Error Distribution")

        plt.title(config.get('title', 'Histogram'), fontsize=16)
        plt.xlabel(config.get('xlabel', 'Absolute Error'), fontsize=12)
        plt.ylabel(config.get('ylabel', 'Count'), fontsize=12)
        
        if 'xlim' in config: plt.xlim(config['xlim'])
        if 'ylim' in config: plt.ylim(config['ylim'])

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.tight_layout()


# ==============================================================================
#  Main Execution 
# ==============================================================================
if __name__ == "__main__":
    # 1. Configuration 
    # ----------------------------------------------------
    # 
    QUANT_SCHEME_TO_ANALYZE = 'per-group'
    QUANT_KEY_TO_ANALYZE = 'int4-asym'
    TENSOR_NAME = 'model.decoder.layers.0.self_attn.k_proj.weight'
    ANALYSIS_RANGE = (0, 127)
    
    ANALYSIS_PLOTTING = True 
    ENABLE_PLOTTING = True
    
    # Select grpah type
        # 'elementwise', 'absolute', 'histogram', 'all
    PLOT_TYPE_TO_GENERATE = 'absolute'

    # 2. Plotting Configuration 
    # --------------------------------------------------
    PLOT_CONFIG = {
        'elementwise_error_plot': {
            'title': f'Element-wise Error ({QUANT_SCHEME_TO_ANALYZE} {QUANT_KEY_TO_ANALYZE})',
            'xlabel': 'Element ID',
            'ylabel': 'Quantization Error',
            'xlim': (0, 127),         
            'ylim': (-0.0005, 0.0005),     
            'xticks_major': 10,      
            'yticks_major': 0.000125,     
            'linewidth': 2.0,          
            'color': 'dodgerblue'      
        },
        'abs_error_plot': {
            'title': f'Absolute Element-wise Error ({QUANT_SCHEME_TO_ANALYZE} {QUANT_KEY_TO_ANALYZE})',
            'xlabel': 'Element ID',
            'ylabel': 'Absolute Quantization Error',
            'xlim': (0, 127),
            'ylim': (0, 0.0005),
            'xticks_major': 10,
            'yticks_major': 0.000125,
            'linewidth': 2.0,
            'color': 'orangered'
        },
        'histogram_plot': {
            'title': f'Absolute Error Distribution ({QUANT_SCHEME_TO_ANALYZE} {QUANT_KEY_TO_ANALYZE})',
            'xlabel': 'Absolute Error',
            'ylabel': 'Element Count',
            'bins': 200,               
            'xlim': (0, 0.01),        
            'color': 'forestgreen'
        }
    }

    # 3. Analysis Execution 
    # ----------------------------------------
    try:
        analyzer = QuantizationErrorAnalyzer(
            quant_scheme=QUANT_SCHEME_TO_ANALYZE,
            quant_key=QUANT_KEY_TO_ANALYZE,
            tensor_name=TENSOR_NAME,
            analysis_range = ANALYSIS_RANGE 
        )
        
        analyzer.print_statistics()
        
        if ENABLE_PLOTTING:
            analyzer.generate_plots(PLOT_CONFIG, plot_type=PLOT_TYPE_TO_GENERATE)
            
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"\n[error] analysis is aborted {e}")