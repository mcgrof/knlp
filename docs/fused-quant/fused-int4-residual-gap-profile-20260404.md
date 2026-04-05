# Fused INT4 residual-gap profile run (2026-04-04)

## Run root
- 

## Summary
The profiling run attributed the residual decode gap to fused-only decode/cache kernels rather than shared model kernels or graph/runtime asymmetry.

## Key takeaways
- graph restoration fixed the major earlier bug
- remaining gap appears inside  and supporting cache/write ops
- stage profiling scripts used for this run are preserved under 

## Preserved scripts
- 
- 
- 
- 
- 
- 
