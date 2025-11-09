<script lang="ts">
	interface Props {
		runs: any[];
	}

	let { runs }: Props = $props();

	let hoveredId = $state<number | null>(null);

	function getStatusColor(status: string) {
		switch (status) {
			case 'completed':
				return 'bg-green-100 text-green-800';
			case 'running':
				return 'bg-blue-100 text-blue-800';
			case 'failed':
				return 'bg-red-100 text-red-800';
			default:
				return 'bg-gray-100 text-gray-800';
		}
	}

	function getRunTypeIcon(type: string) {
		return type === 'training' ? '⚙️' : '↗️';
	}
</script>

<div class="bg-white shadow rounded-lg">
	<div class="px-6 py-4 border-gray-200 border-b">
		<h3 class="font-semibold text-gray-900 text-lg">Runs ({runs.length})</h3>
	</div>

	<div class="divide-y divide-gray-200 max-h-96 overflow-y-auto">
		{#if runs.length === 0}
			<div class="px-6 py-8 text-gray-500 text-center">No runs yet. Start training or inference to create runs.</div>
		{:else}
			{#each runs as run}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div class="hover:bg-gray-50 px-6 py-3 transition-colors" onmouseenter={() => (hoveredId = run.id)} onmouseleave={() => (hoveredId = null)}>
					<div class="flex justify-between items-start">
						<div class="flex-1 min-w-0">
							<div class="flex items-center gap-2">
								<span class="text-lg">{getRunTypeIcon(run.run_type)}</span>
								<p class="font-medium text-gray-900 text-sm">
									{run.run_type === 'training' ? 'Training' : 'Inference'} #{run.id}
								</p>
								<span class="px-2 py-0.5 text-xs rounded-full {getStatusColor(run.status)}">
									{run.status}
								</span>
							</div>

							<div class="mt-1 text-gray-500 text-xs">
								<p>Started: {new Date(run.created_at).toLocaleString()}</p>
								{#if run.completed_at}
									<p>Completed: {new Date(run.completed_at).toLocaleString()}</p>
								{/if}
							</div>
						</div>
					</div>

					<!-- {#if hoveredId === run.id} -->
					{#if true}
						<div class="bg-gray-50 mt-2 p-2 rounded text-xs">
							{#if run.meta}
								<p class="mb-1 text-gray-600">
									<strong>Config:</strong>
									{JSON.stringify(run.meta)}
								</p>
							{/if}
							{#if run.result}
								<p class="text-gray-600">
									<strong>Result:</strong>
									{JSON.stringify(run.result)}
								</p>
							{/if}
						</div>
					{/if}
				</div>
			{/each}
		{/if}
	</div>
</div>
