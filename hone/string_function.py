import textgrad as tg
from textgrad.autograd.function import BackwardContext


class GeneralStringFunction(tg.autograd.StringBasedFunction):
    def __init__(self, fn, function_purpose):
        super().__init__(
            fn=fn,
            function_purpose=function_purpose
        )

    def forward(self, *args, **kwargs) -> tg.Variable:
        # Create a dictionary of inputs with string keys
        all_inputs = {
            f"arg_{i}": arg for i, arg in enumerate(args)
        }
        all_inputs.update(kwargs)
        
        # Get result using the function
        result = self.fn(*args, **kwargs)

        # Create response variable with proper context
        response = tg.Variable(
            value=result,
            predecessors=list(all_inputs.values()),
            role_description=f"Output of the string-based function with purpose: {self.function_purpose}",
            requires_grad=True
        )

        # Set gradient function
        response.set_grad_fn(BackwardContext(
            backward_fn=self.backward,
            response=response,
            function_purpose=self.function_purpose,
            inputs=all_inputs
        ))

        return response