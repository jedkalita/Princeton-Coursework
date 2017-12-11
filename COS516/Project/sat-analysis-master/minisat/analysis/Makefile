EXEC      = analyze
DEPDIR    = mtl utils

include $(MROOT)/mtl/template.mk

BENCHDIR = ../../bench/simple

BENCHMARKS = one two three contradict easy-sat imply-sat sparse-sat 

ifneq ("$(wildcard ../core/minisat_debug)","")
SOLVER = ../core/minisat_debug
else
SOLVER = ../core/minisat_static
endif

cleanoutput:
	-rm *.trace
	-rm *.csv

clean-all: clean cleanoutput

%.trace: $(BENCHDIR)/%.cnf
	-$(SOLVER) -verb=0 $< $@

%.csv: %.trace d
	./analyze_debug $< $@

run-simple: $(addsuffix .trace, $(BENCHMARKS))

analyze-simple: $(addsuffix .csv, $(BENCHMARKS))
