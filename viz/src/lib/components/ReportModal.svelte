<script lang="ts">
	import { onMount } from 'svelte';
	import axios from 'axios';

	interface Props {
		API_URL: string;
		model: any;
		onClose: () => void;
	}

	let { API_URL, model, onClose }: Props = $props();

	let report = $state<any>(null);
	let loading = $state(true);

	onMount(async () => {
		try {
			const response = await axios.get(`${API_URL}/scenarios/report/${model.id}`);
			report = response.data;
		} catch (error) {
			console.error('Error fetching report:', error);
		} finally {
			loading = false;
		}
	});

	function openInNewWindow() {
		const reportWindow = window.open('', '_blank', 'width=800,height=600');
		if (reportWindow && report) {
			reportWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>Model Report - ${report.model.name}</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #1f2937; }
            h2 { color: #374151; margin-top: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f3f4f6; }
            .metric { display: inline-block; margin: 10px 20px 10px 0; }
            .metric-label { font-weight: bold; color: #6b7280; }
            .metric-value { font-size: 1.2em; color: #1f2937; }
          </style>
        </head>
        <body>
          <h1>Model Report: ${report.model.name}</h1>
          
          <h2>Model Information</h2>
          <div class="metric">
            <div class="metric-label">Type</div>
            <div class="metric-value">${report.model.type}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Version</div>
            <div class="metric-value">${report.model.version}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Created</div>
            <div class="metric-value">${new Date(report.model.created_at).toLocaleString()}</div>
          </div>
          
          <h2>Performance Metrics</h2>
          <div class="metric">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">${(report.performance.accuracy * 100).toFixed(2)}%</div>
          </div>
          <div class="metric">
            <div class="metric-label">Train Size</div>
            <div class="metric-value">${report.performance.train_size}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Test Size</div>
            <div class="metric-value">${report.performance.test_size}</div>
          </div>
          
          <h2>Hyperparameters</h2>
          <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            ${Object.entries(report.hyperparameters)
				.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
				.join('')}
          </table>
          
          ${
				report.feature_importance && Object.keys(report.feature_importance).length > 0
					? `
            <h2>Feature Importance</h2>
            <table>
              <tr><th>Feature</th><th>Importance</th></tr>
              ${Object.entries(report.feature_importance)
					.sort((a: any, b: any) => b[1] - a[1])
					.map(([k, v]: any) => `<tr><td>${k}</td><td>${v.toFixed(4)}</td></tr>`)
					.join('')}
            </table>
          `
					: ''
			}
          
          <h2>Classification Report</h2>
          <table>
            <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
            ${Object.entries(report.classification_report)
				.filter(([k]) => !['accuracy', 'macro avg', 'weighted avg'].includes(k))
				.map(
					([cls, metrics]: any) => `
                <tr>
                  <td>${cls}</td>
                  <td>${metrics.precision?.toFixed(3) || 'N/A'}</td>
                  <td>${metrics.recall?.toFixed(3) || 'N/A'}</td>
                  <td>${metrics['f1-score']?.toFixed(3) || 'N/A'}</td>
                  <td>${metrics.support || 'N/A'}</td>
                </tr>
              `
				)
				.join('')}
          </table>
          
          ${
				report.confusion_matrix
					? `
            <h2>Confusion Matrix</h2>
            <table>
              ${report.confusion_matrix
					.map(
						(row: any) => `
                <tr>${row.map((cell: any) => `<td>${cell}</td>`).join('')}</tr>
              `
					)
					.join('')}
            </table>
          `
					: ''
			}
        </body>
        </html>
      `);
			reportWindow.document.close();
		}
	}
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="z-50 fixed inset-0 flex justify-center items-center bg-black/50" onclick={onClose}>
	<div class="bg-white shadow-xl m-4 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-y-auto" onclick={(e) => e.stopPropagation()}>
		<div class="top-0 sticky flex justify-between items-center bg-white px-6 py-4 border-gray-200 border-b">
			<h2 class="font-bold text-gray-900 text-2xl">Model Report</h2>
			<div class="flex gap-2">
				<button onclick={openInNewWindow} class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-white text-sm transition-colors">
					Open in New Window
				</button>
				<button onclick={onClose} class="bg-gray-200 hover:bg-gray-300 px-4 py-2 rounded text-gray-700 text-sm transition-colors"> Close </button>
			</div>
		</div>

		<div class="px-6 py-6">
			{#if loading}
				<div class="flex justify-center items-center py-12">
					<div class="border-4 border-t-transparent border-blue-600 rounded-full w-12 h-12 animate-spin"></div>
				</div>
			{:else if report}
				<div class="space-y-6">
					<!-- Model Info -->
					<section>
						<h3 class="mb-3 font-semibold text-gray-900 text-lg">Model Information</h3>
						<div class="gap-4 grid grid-cols-2 md:grid-cols-4">
							<div class="bg-gray-50 p-3 rounded">
								<p class="mb-1 text-gray-600 text-xs">Name</p>
								<p class="font-medium text-gray-900">{report.model.name}</p>
							</div>
							<div class="bg-gray-50 p-3 rounded">
								<p class="mb-1 text-gray-600 text-xs">Type</p>
								<p class="font-medium text-gray-900">{report.model.type}</p>
							</div>
							<div class="bg-gray-50 p-3 rounded">
								<p class="mb-1 text-gray-600 text-xs">Version</p>
								<p class="font-medium text-gray-900">{report.model.version}</p>
							</div>
							<div class="bg-gray-50 p-3 rounded">
								<p class="mb-1 text-gray-600 text-xs">Created</p>
								<p class="font-medium text-gray-900 text-xs">
									{new Date(report.model.created_at).toLocaleDateString()}
								</p>
							</div>
						</div>
					</section>

					<!-- Performance -->
					<section>
						<h3 class="mb-3 font-semibold text-gray-900 text-lg">Performance Metrics</h3>
						<div class="gap-4 grid grid-cols-3">
							<div class="bg-green-50 p-4 rounded">
								<p class="mb-1 text-green-600 text-sm">Accuracy</p>
								<p class="font-bold text-green-900 text-3xl">
									{(report.performance.accuracy * 100).toFixed(2)}%
								</p>
							</div>
							<div class="bg-blue-50 p-4 rounded">
								<p class="mb-1 text-blue-600 text-sm">Train Size</p>
								<p class="font-bold text-blue-900 text-3xl">{report.performance.train_size}</p>
							</div>
							<div class="bg-purple-50 p-4 rounded">
								<p class="mb-1 text-purple-600 text-sm">Test Size</p>
								<p class="font-bold text-purple-900 text-3xl">{report.performance.test_size}</p>
							</div>
						</div>
					</section>

					<!-- Hyperparameters -->
					<section>
						<h3 class="mb-3 font-semibold text-gray-900 text-lg">Hyperparameters</h3>
						<div class="bg-gray-50 p-4 rounded">
							<pre class="text-sm">{JSON.stringify(report.hyperparameters, null, 2)}</pre>
						</div>
					</section>

					<!-- Feature Importance -->
					{#if report.feature_importance && Object.keys(report.feature_importance).length > 0}
						<section>
							<h3 class="mb-3 font-semibold text-gray-900 text-lg">Feature Importance</h3>
							<div class="space-y-2">
								{#each Object.entries(report.feature_importance).sort((a: any, b: any) => b[1] - a[1]) as [feature, importance]}
									<div class="flex items-center gap-3">
										<span class="w-24 text-gray-700 text-sm">{feature}</span>
										<div class="flex-1 bg-gray-200 rounded-full h-4">
											<div class="bg-blue-600 rounded-full h-4" style="width: {((importance as number) * 100).toFixed(1)}%"></div>
										</div>
										<span class="w-16 text-gray-600 text-sm text-right">
											{(importance as number).toFixed(4)}
										</span>
									</div>
								{/each}
							</div>
						</section>
					{/if}

					<!-- Classification Report -->
					<section>
						<h3 class="mb-3 font-semibold text-gray-900 text-lg">Classification Report</h3>
						<div class="overflow-x-auto">
							<table class="w-full text-sm">
								<thead class="bg-gray-100">
									<tr>
										<th class="px-4 py-2 text-left">Class</th>
										<th class="px-4 py-2 text-right">Precision</th>
										<th class="px-4 py-2 text-right">Recall</th>
										<th class="px-4 py-2 text-right">F1-Score</th>
										<th class="px-4 py-2 text-right">Support</th>
									</tr>
								</thead>
								<tbody class="divide-y divide-gray-200">
									{#each Object.entries(report.classification_report) as [cls, metrics]}
										{#if !['accuracy', 'macro avg', 'weighted avg'].includes(cls)}
											<tr>
												<td class="px-4 py-2 font-medium">{cls}</td>
												<td class="px-4 py-2 text-right">{metrics.precision?.toFixed(3) || 'N/A'}</td>
												<td class="px-4 py-2 text-right">{metrics.recall?.toFixed(3) || 'N/A'}</td>
												<td class="px-4 py-2 text-right">{metrics['f1-score']?.toFixed(3) || 'N/A'}</td>
												<td class="px-4 py-2 text-right">{metrics.support || 'N/A'}</td>
											</tr>
										{/if}
									{/each}
								</tbody>
							</table>
						</div>
					</section>

					<!-- Confusion Matrix -->
					{#if report.confusion_matrix}
						<section>
							<h3 class="mb-3 font-semibold text-gray-900 text-lg">Confusion Matrix</h3>
							<div class="overflow-x-auto">
								<table class="mx-auto text-sm">
									<tbody>
										{#each report.confusion_matrix as row}
											<tr>
												{#each row as cell}
													<td class="px-4 py-2 border border-gray-300 font-medium text-center">
														{cell}
													</td>
												{/each}
											</tr>
										{/each}
									</tbody>
								</table>
							</div>
						</section>
					{/if}
				</div>
			{/if}
		</div>
	</div>
</div>
