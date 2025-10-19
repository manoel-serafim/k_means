INC_DIR := ../include
SRC_DIR := ../source
BUILD_DIR := .
HEAD_DIR := ..
BIN_DIR := $(BUILD_DIR)/bin
OBJ_DIR := $(BUILD_DIR)/obj


SRCS := $(shell find $(SRC_DIR) -name "*.c")
HEADERS := $(wildcard $(INC_DIR)/*.h) $(wildcard $(INC_DIR)/*/*.h)
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

CC := clang
CFLAGS := -Wall -Wextra -Werror -pedantic -O3 -I$(INC_DIR) \
          -Waddress -Warray-bounds -Wcast-align -Wconversion \
          -Wfloat-equal -Wformat -Wimplicit-function-declaration \
          -Wmissing-prototypes -Wnull-dereference -Wshadow \
          -Wsign-compare -Wstrict-aliasing -Wuninitialized \
          -Wunused-variable

TARGET := $(BIN_DIR)/sequential
LDFLAGS := -lm

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# --- Code formatting with uncrustify ---
CRUST := $(patsubst $(SRC_DIR)/%.c, $(SRC_DIR)/%.uncrustify, $(SRCS))
CRUST += $(patsubst $(INC_DIR)/%.h, $(INC_DIR)/%.uncrustify, $(HEADERS))


uncrustify: $(CRUST)

../%.uncrustify: ../%.c
	uncrustify -c ../.uncrustify.cfg -f $< > $@

../%.uncrustify: ../%.h
	uncrustify -c ../.uncrustify.cfg -f $< > $@

# --- Static analysis with cppcheck ---
cppcheck:
	cppcheck --force -q --check-level=exhaustive --enable=all \
		$(SRCS) $(HEADERS) -I $(INC_DIR) \
		--checkers-report=cppcheck_report.xml

.PHONY: all clean uncrustify cppcheck
