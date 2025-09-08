import highspy
import logging

# Python logging (for your messages)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

h = highspy.Highs()

# Enable HiGHS solver output to console
h.setOptionValue("output_flag", True)
h.setOptionValue("log_to_console", True)
h.setOptionValue("print_level", 2)

log.info("Starting")

filename = 'Scenario_1_model.mps'
h.readModel(filename)
log.info(f"Model {filename} read successfully")

log.info("Starting optimization run")
h.run()
log.info("Optimization run complete")

model_status = h.getModelStatus()
log.info(f"Model {filename} has status {model_status}")

# gurobi sol in 1140s 