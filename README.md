# Spatial Statistics for Positional Voting Rules

Code and data for the paper. Each folder is one experiment: `run.py` computes the results and writes them to `data/`, and `plot.py` draws the paper's figures from `data/` into `figures/`. Shared code is in `common/`.

```bash
pip install -r requirements.txt
cd seats_votes && python run.py && python plot.py
```

| Folder | In the paper |
|---|---|
| `borda_surfaces` | Figures 1 and 2 (Section 3.4) |
| `borda_winner` | Figure 3 (Section 5.1) |
| `plurality_boundary` | Figure 4 (Section 5.2) and Figure 10 (Appendix A.4) |
| `score_profiles` | Figure 5 (Section 5.3) |
| `seats_votes` | Figure 6 (Section 5.4) |
| `winner_density` | Figure 7 (Section 5.5) |
| `representativeness` | Figures 8 and 9 (Section 5.6) |
| `borda_moments` | Figures 11 and 12 (Appendix A.5) |
| `borda_checks` | Numerical checks in Appendices A.3 and B.3 (no figure) |
| `reflection_diagram` | Figure 13 (Appendix B.1; a diagram, no data) |
