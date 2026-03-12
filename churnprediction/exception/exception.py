import sys
from churnprediction.logging import logger


class ChurnPredictionException(Exception):

    def __init__(self, error_message, error_detail):
        super().__init__(error_message)

        _, _, exc_tb = error_detail.exc_info()

        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.lineno = exc_tb.tb_lineno
        self.error_message = error_message

        logger.error(self.__str__())

    def __str__(self):
        return (
            f"Error occurred in python script [{self.file_name}] "
            f"at line [{self.lineno}] "
            f"error message [{self.error_message}]"
        )