#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import warnings
import ldm.invoke.CLI
warnings.warn("dream.py is being deprecated, please run invoke.py for the "
              "new UI/API or legacy_api.py for the old API",
              DeprecationWarning)
ldm.invoke.CLI.main()

