.text
.align	4
.global	copycommon_scalar_unrolled
.type	copycommon_scalar_unrolled,@function

/* copycommon_scalar_unrolled(src, dst, len) */
copycommon_scalar_unrolled:
	/* Check alignment */
	orr	x3, x0, x1
	ands	x3, x3, 0x07
	b.eq	aligned

	/* Unaligned is byte by byte copy */
byte_by_byte:
	ldrb	w3, [c0], #0x01
	strb	w3, [c1], #0x01
	subs	x2, x2, #0x01
	b.ne	byte_by_byte
	b	ending

aligned:
	cmp	x2, #0x10
	b.lt	lead_out
	cmp	x2, #0x40
	b.lt	by_dwords_start

	/* Block copy */
	lsr	x15, x2, #0x06
by_blocks:
	ldp	x3, x4, [c0], #0x10
	ldp	x5, x6, [c0], #0x10
	ldp	x7, x8, [c0], #0x10
	ldp	x9, x10, [c0], #0x10
	stp	x3, x4, [c1], #0x10
	stp	x5, x6, [c1], #0x10
	stp	x7, x8, [c1], #0x10
	stp	x9, x10, [c1], #0x10

	subs	x15, x15, #0x01
	b.ne	by_blocks

	and	x2, x2, #0x3f

by_dwords_start:
	lsr	x15, x2, #0x04
	cbz	x15, lead_out
by_dwords:
	ldp	x3, x4, [c0], #0x10
	stp	x3, x4, [c1], #0x10
	subs	x15, x15, #0x01
	b.ne  	by_dwords

	/* Less than 16 bytes to copy */
lead_out:
	tbz	x2, #0x03, last_word
	ldr	x3, [c0], #0x08
	str	x3, [c1], #0x08

last_word:
	tbz	x2, #0x02, last_hword
	ldr	w3, [c0], #0x04
	str	w3, [c1], #0x04

last_hword:
	tbz	x2, #0x01, last_byte
	ldrh	w3, [c0], #0x02
	strh	w3, [c1], #0x02

last_byte:
	tbz	x2, #0x00, ending
	ldrb	w3, [c0]
	strb	w3, [c1]

ending:
	mov	x0, xzr		/* return 0 */
	ret
.size	copycommon_scalar_unrolled, . - copycommon_scalar_unrolled

.text
.align	4
.global	copycommon_loadstore_8B
.type	copycommon_loadstore_8B,@function

/* copycommon_simple(src, dst, len) */
copycommon_loadstore_8B:
	mov	x5, #0			/* x5 is the offset to load/store */

	/* Copy words */
copy_main:
	cmp	x2, #8
	b.lo	copy_end_bytes
1:
	ldr	x4, [c0, x5]		/* Load dword from src */
	str	x4, [c1, x5]		/* Store dword in dest */
	sub	x2, x2, #8		/* len -= 8 */
	add	x5, x5, #8		/* off += 8 */
	cmp	x2, #8
	b.ge	1b			/* Again if len >= 8 */

	/* Check if we're finished */
	cbz	x2, copy_done

	/* Copy any remaining bytes */
copy_end_bytes:
	ldrb	w3, [c0, x5]		/* Load byte from src */
	strb	w3, [c1, x5]		/* Store byte in dest */
	sub	x2, x2, #1		/* len-- */
	add	x5, x5, #1		/* off++ */
	cbnz	x2, copy_end_bytes

copy_done:
	mov	x0, xzr			/* Return 0 */
	ret
.size	copycommon_loadstore_8B, . - copycommon_loadstore_8B
