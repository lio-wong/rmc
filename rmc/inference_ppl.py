from rmc import constants, utils
def evaluate_probabilistic_program(program, sampling_budget, inference_timeout=600.0, sampling_method='rejection'):
    try:
        # TODO -- distinguish between error types
        model_str = program.to_string(posterior_samples=sampling_budget, sampling_method=sampling_method)
        print("\n\nrunning model str: ", model_str)
        print("\n---------\n")

        keys, samples = utils.run_webppl(code=model_str, timeout=inference_timeout)
        
        if "ERROR" in samples: 
            return -1, samples, "Code failed"

        if "value" not in eval(samples)[0]:
            # then ended with an error, not the actual samples
            return -1, samples, "Code failed"
        return 1, samples, "Code succeed"
    except Exception as e:
        print("PPL runtime error: ", e)
        return -1, str(e), "Code failed"