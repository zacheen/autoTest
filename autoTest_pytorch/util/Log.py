# including the text and picture
import datetime
from pathlib import Path

# << Time format >>
ISOTIMEFORMAT      = '%Y_%m_%d_%H_%M_%S'   # used in filenames
format_for_db_time = '%Y-%m-%d %H:%M'      # used for DB search timestamps

def get_now_time():
    return datetime.datetime.now().strftime(ISOTIMEFORMAT)

# << Log >>
class Logger():
    # Report output folders (created on import)
    TEST_REPORT_PATH = Path("./testreport")
    TEST_REPORT_PATH.mkdir(parents=True, exist_ok=True)
    TEST_PIC_PATH    = TEST_REPORT_PATH / "testpic"
    TEST_PIC_PATH.mkdir(parents=True, exist_ok=True)
    print("check/make folder successfully")
    
    now_time = get_now_time()
    log_dir = TEST_REPORT_PATH / now_time
    log_dir.mkdir(parents=True, exist_ok=True)
    
    pipe_output_f = open(log_dir / 'pipe_output.txt', "w", encoding='UTF-8')
    cmd_output_f  = open(log_dir / 'cmd_output.txt',  "w", encoding='UTF-8')
    error_f       = open(log_dir / 'error.txt',       "w", encoding='UTF-8')

    def cmd_write(s):
        Logger.cmd_output_f.write(s)
        Logger.cmd_output_f.flush()

    def pipe_write(s):
        Logger.pipe_output_f.write(s)
    
    def error_write(s):
        Logger.error_f.write(s)
    def error_flush():
        Logger.error_f.flush()

from . import HTMLTestRun
def print_to_output(stri):
    """Print to console, HTML report, and cmd_output.txt."""
    print(stri)
    HTMLTestRun.p_to_html(str(stri) + "\n")
    Logger.cmd_write(str(stri) + "\n")

def report_screenshot_path(className, file_create_time):
    return str(Logger.TEST_PIC_PATH / f'{className}_{file_create_time}.png')

def report_error(round_num, why=None):
    Logger.error_write(f"error round : {round_num}\n")
    Logger.error_write(f"error time : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if why is not None:
        Logger.error_write.write(why + "\n")
    Logger.error_flush()

