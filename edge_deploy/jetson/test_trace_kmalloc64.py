import unittest

from trace_kmalloc64 import loss_counts, parse_trace, summarize, symbolize


def event(ts, pointer='ffff000000001000', name='kmalloc', site='ffff000000010010'):
    sizes = '' if name == 'kfree' else ' bytes_req=40 bytes_alloc=64 gfp_flags=GFP_KERNEL'
    return (' nvphsd-6207 [002] d..1 {}: {}: call_site={} ptr={}{}'
            .format(ts, name, site, pointer, sizes))


class TraceTests(unittest.TestCase):
    def rows(self, text):
        rows, errors = parse_trace(text)
        self.assertEqual(errors, 0)
        symbolize(rows, ['ffff000000010000 t example [driver]'])
        return rows

    def test_cross_cpu_sort_and_pointer_lifetime(self):
        rows = self.rows('\n'.join([
            event('12.100002', name='kfree'), event('12.100001'),
            event('12.100003', name='kmalloc_node')]))
        result = summarize(rows)
        self.assertEqual(result['matched_frees'], 1)
        self.assertEqual(result['not_freed_by_end'], 1)
        self.assertEqual(result['pointer_reuse_without_observed_free'], 0)
        self.assertEqual(rows[0]['symbol'], 'example [driver]+0x10')

    def test_boundary_free_is_not_a_negative_allocation(self):
        result = summarize(self.rows(event('12.100001', name='kfree')))
        self.assertEqual(result['frees_without_sampled_allocation'], 1)
        self.assertEqual(result['not_freed_by_end'], 0)

    def test_lost_free_or_reused_pointer_flagged(self):
        result = summarize(self.rows(event('1.001') + '\n' + event('1.002')))
        self.assertEqual(result['pointer_reuse_without_observed_free'], 1)
        self.assertEqual(result['ambiguous_pointer_count'], 1)

    def test_bad_and_unexpected_event_rejected(self):
        rows, rejected = parse_trace('# header\n' + event('1.000').replace(
            'bytes_alloc=64', 'bytes_alloc=128') + '\nmalformed')
        self.assertEqual(rows, [])
        self.assertEqual(rejected, 2)

    def test_null_and_timestamp_tie(self):
        result = summarize(self.rows('\n'.join([
            event('1.000', pointer='0000000000000000'),
            event('1.001'), event('1.001', name='kfree')])))
        self.assertEqual(result['null_allocations'], 1)
        self.assertEqual(result['ambiguous_pointer_count'], 1)
        self.assertEqual(result['groups'][0]['allocations'], 1)

    def test_no_kernel_addresses_cannot_be_silently_resolved(self):
        with self.assertRaises(ValueError):
            symbolize([], ['0000000000000000 t hidden'])

    def test_loss_counter_validation(self):
        self.assertEqual(loss_counts({'cpu0': 'entries: 4\noverrun: 3\n'
                                     'commit overrun: 0\ndropped events: 1'}),
                         {'cpu0': {'overrun': 3, 'commit overrun': 0, 'dropped events': 1}})
        with self.assertRaises(KeyError):
            loss_counts({'cpu0': 'entries: 4'})


if __name__ == '__main__':
    unittest.main()
