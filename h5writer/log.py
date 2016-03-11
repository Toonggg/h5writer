import time
t0 = time.time()

import logging,inspect
log_and_raise_error = lambda logger, message: log(logger, message, lvl="ERROR", exception=RuntimeError, rollback=2)
log_warning = lambda logger, message: log(logger, message, lvl="WARNING", exception=None, rollback=2)
log_info = lambda logger, message: log(logger, message, lvl="INFO", exception=None, rollback=2)
log_debug = lambda logger, message: log(logger, message, lvl="DEBUG", exception=None, rollback=2)


def log(logger, message, lvl, exception=None, rollback=1):
    logcalls = {"ERROR": logger.error,
                "WARNING": logger.warning,
                "DEBUG": logger.debug,
                "INFO": logger.info}
    if lvl not in logcalls:
        print "%s is an invalid logger level." % lvl
        sys.exit(1)
    logcall = logcalls[lvl]
    # This should maybe go into a handler
    if (logger.getEffectiveLevel() >= logging.INFO) or rollback is None:
        # Short output
        msg = "%s" % message
    else:
        # Detailed output only in debug mode
        func = inspect.currentframe()
        for r in range(rollback):
            # Rolling back in the stack, otherwise it would be this function
            func = func.f_back
        code = func.f_code
        #msg = "%s\n\t=> in \'%s\' function \'%s\' [%s:%i]" % (message,
        #                                                      func.f_globals["__name__"],
        #                                                      code.co_name, 
        #                                                      code.co_filename, 
        #                                                      code.co_firstlineno)
        msg = message
        
    logcall("\t%i sec\t%s" % (time.time()-t0, msg))
    if exception is not None:
        raise exception(message)
