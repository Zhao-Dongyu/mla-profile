import pandas as pd
import matplotlib.pyplot as plt

def plot_heatmap_from_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    
    df['Relative'] = df['FA3'] / df['FlashMLA'] * 100
    
    nheads = df['nhead'].unique()
    
    for nhead in nheads:
        df_nhead = df[df['nhead'] == nhead]
        
        df_pivot = df_nhead.pivot(index="bs", columns="kv_len", values="Relative")
        
        plt.figure(figsize=(10, 6))
        plt.imshow(df_pivot, aspect="auto", cmap="coolwarm_r", origin="lower")
        
        plt.xticks(range(len(df_pivot.columns)), df_pivot.columns, rotation=45)
        plt.yticks(range(len(df_pivot.index)), df_pivot.index)
        plt.xlabel("kv_len")
        plt.ylabel("bs")
        plt.title(f"Relative Performance (%) for nhead={nhead}")
        
        plt.colorbar(label="Relative (%)")
        
        # plt.show()
        plt.savefig(f"heatmap_nhead_{nhead}.png", dpi=300, bbox_inches="tight")
        plt.close()

plot_heatmap_from_file("result_cleaned.txt")
