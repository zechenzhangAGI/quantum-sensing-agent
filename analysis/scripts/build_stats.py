"""Merge stats_logs.json, esr_fit.json and data/guidance_checks.csv into stats.json."""
import os, csv, json
A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = json.load(open(f'{A}/stats_logs.json'))
s['esr_fit_candidate3_14may'] = json.load(open(f'{A}/esr_fit.json'))
g = list(csv.DictReader(open(f'{A}/data/guidance_checks.csv')))
s['parametric_guidance'] = dict(n=len(g), followed_in_next_written_config=sum(r['followed'] == 'True' for r in g),
                                rows=[{k: r[k] for k in ('log', 'line', 'guidance', 'wanted', 'write_line', 'followed')} for r in g])
s['retrieval'] = dict(
    rag_queries_logged_feb_may=5, rag_query_logs=['projects/NVExperiment/runs/run_20250502_123549 (4)', 'projects/NVExperiment/runs/run_20250502_123846 (1)'],
    rag_similarity_range_logged=[0.38, 0.44],
    rag_search_on_hardware=dict(log='projects/NVExperiment/runs/run_20250616_112227 (line 866)', n=1, top_similarity=0.57,
        top_chunk='12 May galvo/find_nv session', effect='none: the next reply asks the operator for ESR settings'),
    embedding_files=9, embedding_vectors_per_file=1, embedding_dim=384,
    note='see the technical report, retrieval sections')
s['earlier_count_correction'] = dict(earlier=79, rederived=72, removed=7,
    reason='7 "[Runner] Saved ... data" lines (Feb-May) are inside LLM responses that invented tool output (data/hallucinated_turns.csv)',
    with_16_june=dict(total=73, invented_saved_lines_total=10, note='16 June adds 1 real galvo scan and 3 invented saved lines'))
json.dump(s, open(f'{A}/stats.json', 'w'), indent=1, default=str)
print('stats.json written;', s['measurements_total'], 'measurements;', s['parametric_guidance']['followed_in_next_written_config'], '/', s['parametric_guidance']['n'])
