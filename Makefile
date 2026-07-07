-include config.mk

NVCC ?= nvcc
AR ?= ar
NVCCFLAGS ?= -O2 -std=c++20 -Xcompiler -fopenmp
CPPFLAGS := -MMD -MP $(addprefix -I,$(shell find src include -type d 2>/dev/null))

BIGMUL := build/lib/libbigmul.a
BIGMUL_O := $(patsubst src/%.cu,build/%.o,$(shell find src/bigmul -name '*.cu'))

BINS := multiply
$(foreach b,$(BINS),$(eval $(b)_O := $(patsubst src/%.cu,build/%.o,$(shell find src/$(b) -name '*.cu'))))

SRCS := $(shell find src -name '*.cu') $(shell find include -name '*.cuh')
COMPDB := build/compile_commands.json

.PHONY: all clean test $(BINS)

all: $(BINS) $(COMPDB)

$(BINS): %: build/bin/%

test: multiply
	@script/test ./build/bin/multiply

clean:
	@rm -rf build

$(COMPDB): $(SRCS)
	@bear --output $@ -- $(MAKE) --always-make $(BINS) 2>/dev/null; true

$(BIGMUL): $(BIGMUL_O)
	@mkdir -p $(@D)
	@echo "AR   $@"
	@$(AR) rcs $@ $^

.SECONDEXPANSION:
build/bin/%: $$($$*_O) $(BIGMUL)
	@mkdir -p $(@D)
	@echo "LD   $@"
	@$(NVCC) $(NVCCFLAGS) $($*_O) -Lbuild/lib -lbigmul -o $@

build/%.o: src/%.cu
	@mkdir -p $(@D)
	@echo "NVCC $<"
	@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

-include $(BIGMUL_O:.o=.d) $(foreach b,$(BINS),$($(b)_O:.o=.d))
