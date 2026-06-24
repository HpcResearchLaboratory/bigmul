-include config.mk

NVCC      ?= nvcc
AR        ?= ar
NVCCFLAGS ?= -O2

CPPFLAGS := -MMD -MP $(addprefix -I,$(shell find src include -type d 2>/dev/null))
PREFIX   ?= /usr/local
BINDIR   ?= $(PREFIX)/bin
LIBDIR   ?= $(PREFIX)/lib
INCDIR   ?= $(PREFIX)/include

MULTIPLY     := build/bin/multiply
MULTIPLY_OBJS = $(patsubst src/%.cu,build/%.o,$(shell find src/multiply -name '*.cu'))

TEST         := build/bin/test
TEST_OBJS     = $(patsubst src/%.cu,build/%.o,$(shell find src/test -name '*.cu'))

BENCH        := build/bin/bench
BENCH_OBJS    = $(patsubst src/%.cu,build/%.o,$(shell find src/bench -name '*.cu'))

BIGMUL       := build/lib/libbigmul.a
BIGMUL_OBJS   = $(patsubst src/%.cu,build/%.o,$(shell find src/bigmul -name '*.cu'))

.PHONY: all clean run test bench install uninstall compdb

all: $(MULTIPLY)

run: $(MULTIPLY)
	@./$< $(ARGS)

test: $(TEST)
	@./$<

bench: $(BENCH)
	@./$< $(ARGS)

clean:
	@rm -rf build

compdb:
	@mkdir -p build
	@bear --output build/compile_commands.json -- $(MAKE) all

install: $(MULTIPLY) $(BIGMUL)
	@install -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCDIR)/bigmul
	@install -m 755 $(MULTIPLY) $(DESTDIR)$(BINDIR)/
	@install -m 644 $(BIGMUL) $(DESTDIR)$(LIBDIR)/
	@install -m 644 include/bigmul/*.cuh $(DESTDIR)$(INCDIR)/bigmul/

uninstall:
	@rm -f $(DESTDIR)$(BINDIR)/multiply
	@rm -f $(DESTDIR)$(LIBDIR)/libbigmul.a
	@rm -rf $(DESTDIR)$(INCDIR)/bigmul

$(MULTIPLY): $(MULTIPLY_OBJS) $(BIGMUL)
	@mkdir -p $(@D)
	@echo "LD   $@"
	@$(NVCC) $(NVCCFLAGS) $(MULTIPLY_OBJS) -Lbuild/lib -lbigmul -o $@

$(TEST): $(TEST_OBJS) $(BIGMUL)
	@mkdir -p $(@D)
	@echo "LD   $@"
	@$(NVCC) $(NVCCFLAGS) $(TEST_OBJS) -Lbuild/lib -lbigmul -o $@

$(BENCH): $(BENCH_OBJS) $(BIGMUL)
	@mkdir -p $(@D)
	@echo "LD   $@"
	@$(NVCC) $(NVCCFLAGS) $(BENCH_OBJS) -Lbuild/lib -lbigmul -o $@

$(BIGMUL): $(BIGMUL_OBJS)
	@mkdir -p $(@D)
	@echo "AR   $@"
	@$(AR) rcs $@ $^

build/%.o: src/%.cu
	@mkdir -p $(@D)
	@echo "NVCC $<"
	@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

-include $(MULTIPLY_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(BENCH_OBJS:.o=.d) $(BIGMUL_OBJS:.o=.d)
