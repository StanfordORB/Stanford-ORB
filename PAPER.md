### Qualitative Figures

To make qualitative figures, adjust keyword arguments for `plot_scene` in the following script and run it:
```bash
python scripts/paper/qualitative.py
```

### LaTex Tables
To make LaTex tables, run the following script:
```bash
python scripts/paper/leaderboard_to_table.py
```
You could toggle the boolean flag `MAIN` in the script above to determine whether standard deviations are included in the table.

Running the script above produces the following console outputs:
```
...output omitted...
Final:
======
& Depth$\downarrow$ & Normal$\downarrow$ & Shape$\downarrow$ & PSNR-H$\uparrow$ & PSNR-L$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & PSNR-H$\uparrow$ & PSNR-L$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$\\\midrule
...output omitted...
````
Then copy all console outputs after `Final:\n======`) to the following template to produce a LaTex table:

```latex
\begin{table}[t]
\centering
\setlength{\tabcolsep}{3pt}
\scriptsize
\caption{\textbf{Benchmark Comparison of Existing Methods}.
\textdagger denotes models trained with the ground-truth 3D scans and pseudo materials optimized from light-box captures. 
Depth SI-MSE $\times 10^{-3}$. Shape Chamfer distance $\times 10^{-3}$.
}
\label{tab:benchmark}
\resizebox{1.0\linewidth}{!}{
\begin{tabular}{lccccccccccc}
\toprule
 \multirow{2}{*}{}  
 & \multicolumn{3}{c}{Geometry}
 & \multicolumn{4}{c}{Novel Scene Relighting}            
 & \multicolumn{4}{c}{Novel View Synthesis} 
 \\
 \cmidrule(l){2-4} \cmidrule(l){5-8} \cmidrule(l){9-12}
{#### YOUR CONSOLE OUTPUT HERE ####}
\bottomrule
\end{tabular}
}
\end{table}
```
