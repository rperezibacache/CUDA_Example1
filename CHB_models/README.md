# Test 3
This example run a second order system recursively and then from python si acquire this result and plot.

- compile 
```bash
nvcc -arch=sm_89 -O2 CHB.cu -lcufft -o CHB
```
- run 
```bash
./CHB
```
- python plot 
```bash
python3 plot.py
```

