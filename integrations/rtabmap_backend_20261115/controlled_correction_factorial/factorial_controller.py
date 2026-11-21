"""Process-local V35 factorial adapter. Original modules/defaults are untouched."""
from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35

class ForecastSwitch:
    def __init__(self, original, enabled):
        self.original=original
        self.enabled=bool(enabled)
    def __getattr__(self,name):return getattr(self.original,name)
    def select(self,*args,**kwargs):
        kwargs['geometry_feedback']=self.enabled
        return self.original.select(*args,**kwargs)

class FactorialController(CPUFourModuleControllerV35):
    def __init__(self,*args,observed_correction,forecast_diagnostic,**kwargs):
        if type(observed_correction)is not bool or type(forecast_diagnostic)is not bool:
            raise ValueError('Explicit Boolean factorial levels required')
        if 'mode'in kwargs:raise ValueError('Wrong category interpretation fixed for factorial')
        # Original X/Xnf already differ in measured correction and share the same wrong cue.
        # Delegate measured correction to that existing original mode unchanged.
        super().__init__(*args,mode='swapped'if observed_correction else'swapped_no_feedback',**kwargs)
        # Override only original planner's existing diagnostic switch, never its math/return rule.
        self.planner=ForecastSwitch(self.planner,forecast_diagnostic)
        self.factorial_levels=dict(observed_correction=observed_correction,forecast_diagnostic=forecast_diagnostic)
