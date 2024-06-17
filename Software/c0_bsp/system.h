/*
 * system.h - SOPC Builder system and BSP software package information
 *
 * Machine generated for CPU 'c0_nios2' in SOPC Builder design 'nios_multicore'
 * SOPC Builder design path: C:/Users/pedro/OneDrive/Documentos/CNN-paralel/Quartus/nios_multicore.sopcinfo
 *
 * Generated: Wed Jun 12 10:31:34 BRT 2024
 */

/*
 * DO NOT MODIFY THIS FILE
 *
 * Changing this file will have subtle consequences
 * which will almost certainly lead to a nonfunctioning
 * system. If you do modify this file, be aware that your
 * changes will be overwritten and lost when this file
 * is generated again.
 *
 * DO NOT MODIFY THIS FILE
 */

/*
 * License Agreement
 *
 * Copyright (c) 2008
 * Altera Corporation, San Jose, California, USA.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * This agreement shall be governed in all respects by the laws of the State
 * of California and by the laws of the United States of America.
 */

#ifndef __SYSTEM_H_
#define __SYSTEM_H_

/* Include definitions from linker script generator */
#include "linker.h"


/*
 * CPU configuration
 *
 */

#define ALT_CPU_ARCHITECTURE "altera_nios2_gen2"
#define ALT_CPU_BIG_ENDIAN 0
#define ALT_CPU_BREAK_ADDR 0x00020820
#define ALT_CPU_CPU_ARCH_NIOS2_R1
#define ALT_CPU_CPU_FREQ 50000000u
#define ALT_CPU_CPU_ID_SIZE 1
#define ALT_CPU_CPU_ID_VALUE 0x00000000
#define ALT_CPU_CPU_IMPLEMENTATION "fast"
#define ALT_CPU_DATA_ADDR_WIDTH 0x15
#define ALT_CPU_DCACHE_LINE_SIZE 0
#define ALT_CPU_DCACHE_LINE_SIZE_LOG2 0
#define ALT_CPU_DCACHE_SIZE 0
#define ALT_CPU_EXCEPTION_ADDR 0x00000020
#define ALT_CPU_FLASH_ACCELERATOR_LINES 0
#define ALT_CPU_FLASH_ACCELERATOR_LINE_SIZE 0
#define ALT_CPU_FLUSHDA_SUPPORTED
#define ALT_CPU_FREQ 50000000
#define ALT_CPU_HARDWARE_DIVIDE_PRESENT 0
#define ALT_CPU_HARDWARE_MULTIPLY_PRESENT 1
#define ALT_CPU_HARDWARE_MULX_PRESENT 0
#define ALT_CPU_HAS_DEBUG_CORE 1
#define ALT_CPU_HAS_DEBUG_STUB
#define ALT_CPU_HAS_EXTRA_EXCEPTION_INFO
#define ALT_CPU_HAS_ILLEGAL_INSTRUCTION_EXCEPTION
#define ALT_CPU_HAS_JMPI_INSTRUCTION
#define ALT_CPU_ICACHE_LINE_SIZE 0
#define ALT_CPU_ICACHE_LINE_SIZE_LOG2 0
#define ALT_CPU_ICACHE_SIZE 0
#define ALT_CPU_INST_ADDR_WIDTH 0x12
#define ALT_CPU_NAME "c0_nios2"
#define ALT_CPU_NUM_OF_SHADOW_REG_SETS 0
#define ALT_CPU_OCI_VERSION 1
#define ALT_CPU_RESET_ADDR 0x00000000


/*
 * CPU configuration (with legacy prefix - don't use these anymore)
 *
 */

#define NIOS2_BIG_ENDIAN 0
#define NIOS2_BREAK_ADDR 0x00020820
#define NIOS2_CPU_ARCH_NIOS2_R1
#define NIOS2_CPU_FREQ 50000000u
#define NIOS2_CPU_ID_SIZE 1
#define NIOS2_CPU_ID_VALUE 0x00000000
#define NIOS2_CPU_IMPLEMENTATION "fast"
#define NIOS2_DATA_ADDR_WIDTH 0x15
#define NIOS2_DCACHE_LINE_SIZE 0
#define NIOS2_DCACHE_LINE_SIZE_LOG2 0
#define NIOS2_DCACHE_SIZE 0
#define NIOS2_EXCEPTION_ADDR 0x00000020
#define NIOS2_FLASH_ACCELERATOR_LINES 0
#define NIOS2_FLASH_ACCELERATOR_LINE_SIZE 0
#define NIOS2_FLUSHDA_SUPPORTED
#define NIOS2_HARDWARE_DIVIDE_PRESENT 0
#define NIOS2_HARDWARE_MULTIPLY_PRESENT 1
#define NIOS2_HARDWARE_MULX_PRESENT 0
#define NIOS2_HAS_DEBUG_CORE 1
#define NIOS2_HAS_DEBUG_STUB
#define NIOS2_HAS_EXTRA_EXCEPTION_INFO
#define NIOS2_HAS_ILLEGAL_INSTRUCTION_EXCEPTION
#define NIOS2_HAS_JMPI_INSTRUCTION
#define NIOS2_ICACHE_LINE_SIZE 0
#define NIOS2_ICACHE_LINE_SIZE_LOG2 0
#define NIOS2_ICACHE_SIZE 0
#define NIOS2_INST_ADDR_WIDTH 0x12
#define NIOS2_NUM_OF_SHADOW_REG_SETS 0
#define NIOS2_OCI_VERSION 1
#define NIOS2_RESET_ADDR 0x00000000


/*
 * Define for each module class mastered by the CPU
 *
 */

#define __ALTERA_AVALON_JTAG_UART
#define __ALTERA_AVALON_ONCHIP_MEMORY2
#define __ALTERA_AVALON_SYSID_QSYS
#define __ALTERA_NIOS2_GEN2
#define __CLOCK_COUNTER


/*
 * System configuration
 *
 */

#define ALT_DEVICE_FAMILY "Cyclone IV GX"
#define ALT_ENHANCED_INTERRUPT_API_PRESENT
#define ALT_IRQ_BASE NULL
#define ALT_LOG_PORT "/dev/null"
#define ALT_LOG_PORT_BASE 0x0
#define ALT_LOG_PORT_DEV null
#define ALT_LOG_PORT_TYPE ""
#define ALT_NUM_EXTERNAL_INTERRUPT_CONTROLLERS 0
#define ALT_NUM_INTERNAL_INTERRUPT_CONTROLLERS 1
#define ALT_NUM_INTERRUPT_CONTROLLERS 1
#define ALT_STDERR "/dev/c0_jtag_uart"
#define ALT_STDERR_BASE 0x21010
#define ALT_STDERR_DEV c0_jtag_uart
#define ALT_STDERR_IS_JTAG_UART
#define ALT_STDERR_PRESENT
#define ALT_STDERR_TYPE "altera_avalon_jtag_uart"
#define ALT_STDIN "/dev/c0_jtag_uart"
#define ALT_STDIN_BASE 0x21010
#define ALT_STDIN_DEV c0_jtag_uart
#define ALT_STDIN_IS_JTAG_UART
#define ALT_STDIN_PRESENT
#define ALT_STDIN_TYPE "altera_avalon_jtag_uart"
#define ALT_STDOUT "/dev/c0_jtag_uart"
#define ALT_STDOUT_BASE 0x21010
#define ALT_STDOUT_DEV c0_jtag_uart
#define ALT_STDOUT_IS_JTAG_UART
#define ALT_STDOUT_PRESENT
#define ALT_STDOUT_TYPE "altera_avalon_jtag_uart"
#define ALT_SYSTEM_NAME "nios_multicore"
#define ALT_SYS_CLK_TICKS_PER_SEC NONE_TICKS_PER_SEC
#define ALT_TIMESTAMP_CLK_TIMER_DEVICE_TYPE NONE_TIMER_DEVICE_TYPE


/*
 * altera_hostfs configuration
 *
 */

#define ALTERA_HOSTFS_NAME "/mnt/host"


/*
 * c0_jtag_uart configuration
 *
 */

#define ALT_MODULE_CLASS_c0_jtag_uart altera_avalon_jtag_uart
#define C0_JTAG_UART_BASE 0x21010
#define C0_JTAG_UART_IRQ 0
#define C0_JTAG_UART_IRQ_INTERRUPT_CONTROLLER_ID 0
#define C0_JTAG_UART_NAME "/dev/c0_jtag_uart"
#define C0_JTAG_UART_READ_DEPTH 64
#define C0_JTAG_UART_READ_THRESHOLD 8
#define C0_JTAG_UART_SPAN 8
#define C0_JTAG_UART_TYPE "altera_avalon_jtag_uart"
#define C0_JTAG_UART_WRITE_DEPTH 64
#define C0_JTAG_UART_WRITE_THRESHOLD 8


/*
 * c0_ram configuration
 *
 */

#define ALT_MODULE_CLASS_c0_ram altera_avalon_onchip_memory2
#define C0_RAM_ALLOW_IN_SYSTEM_MEMORY_CONTENT_EDITOR 0
#define C0_RAM_ALLOW_MRAM_SIM_CONTENTS_ONLY_FILE 0
#define C0_RAM_BASE 0x100000
#define C0_RAM_CONTENTS_INFO ""
#define C0_RAM_DUAL_PORT 0
#define C0_RAM_GUI_RAM_BLOCK_TYPE "AUTO"
#define C0_RAM_INIT_CONTENTS_FILE "nios_multicore_c0_ram"
#define C0_RAM_INIT_MEM_CONTENT 0
#define C0_RAM_INSTANCE_ID "NONE"
#define C0_RAM_IRQ -1
#define C0_RAM_IRQ_INTERRUPT_CONTROLLER_ID -1
#define C0_RAM_NAME "/dev/c0_ram"
#define C0_RAM_NON_DEFAULT_INIT_FILE_ENABLED 0
#define C0_RAM_RAM_BLOCK_TYPE "AUTO"
#define C0_RAM_READ_DURING_WRITE_MODE "DONT_CARE"
#define C0_RAM_SINGLE_CLOCK_OP 1
#define C0_RAM_SIZE_MULTIPLE 1
#define C0_RAM_SIZE_VALUE 560000
#define C0_RAM_SPAN 560000
#define C0_RAM_TYPE "altera_avalon_onchip_memory2"
#define C0_RAM_WRITABLE 1


/*
 * c0_rom configuration
 *
 */

#define ALT_MODULE_CLASS_c0_rom altera_avalon_onchip_memory2
#define C0_ROM_ALLOW_IN_SYSTEM_MEMORY_CONTENT_EDITOR 0
#define C0_ROM_ALLOW_MRAM_SIM_CONTENTS_ONLY_FILE 0
#define C0_ROM_BASE 0x0
#define C0_ROM_CONTENTS_INFO ""
#define C0_ROM_DUAL_PORT 1
#define C0_ROM_GUI_RAM_BLOCK_TYPE "AUTO"
#define C0_ROM_INIT_CONTENTS_FILE "nios_multicore_c0_rom"
#define C0_ROM_INIT_MEM_CONTENT 0
#define C0_ROM_INSTANCE_ID "NONE"
#define C0_ROM_IRQ -1
#define C0_ROM_IRQ_INTERRUPT_CONTROLLER_ID -1
#define C0_ROM_NAME "/dev/c0_rom"
#define C0_ROM_NON_DEFAULT_INIT_FILE_ENABLED 0
#define C0_ROM_RAM_BLOCK_TYPE "AUTO"
#define C0_ROM_READ_DURING_WRITE_MODE "DONT_CARE"
#define C0_ROM_SINGLE_CLOCK_OP 1
#define C0_ROM_SIZE_MULTIPLE 1
#define C0_ROM_SIZE_VALUE 90000
#define C0_ROM_SPAN 90000
#define C0_ROM_TYPE "altera_avalon_onchip_memory2"
#define C0_ROM_WRITABLE 1


/*
 * clock_counter_0 configuration
 *
 */

#define ALT_MODULE_CLASS_clock_counter_0 clock_counter
#define CLOCK_COUNTER_0_BASE 0x21000
#define CLOCK_COUNTER_0_IRQ -1
#define CLOCK_COUNTER_0_IRQ_INTERRUPT_CONTROLLER_ID -1
#define CLOCK_COUNTER_0_NAME "/dev/clock_counter_0"
#define CLOCK_COUNTER_0_SPAN 8
#define CLOCK_COUNTER_0_TYPE "clock_counter"


/*
 * hal configuration
 *
 */

#define ALT_INCLUDE_INSTRUCTION_RELATED_EXCEPTION_API
#define ALT_MAX_FD 32
#define ALT_SYS_CLK none
#define ALT_TIMESTAMP_CLK none


/*
 * sysid configuration
 *
 */

#define ALT_MODULE_CLASS_sysid altera_avalon_sysid_qsys
#define SYSID_BASE 0x21008
#define SYSID_ID -559038737
#define SYSID_IRQ -1
#define SYSID_IRQ_INTERRUPT_CONTROLLER_ID -1
#define SYSID_NAME "/dev/sysid"
#define SYSID_SPAN 8
#define SYSID_TIMESTAMP 1718198505
#define SYSID_TYPE "altera_avalon_sysid_qsys"

#endif /* __SYSTEM_H_ */
