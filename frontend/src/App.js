import React, { useState } from 'react';
import { Upload, Users, TrendingUp, Download, BarChart3, PieChart, ChevronsDownUp  } from 'lucide-react';

const App = () => {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [segmentation, setSegmentation] = useState(null);
  const [topCustomers, setTopCustomers] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');

  const API_URL = 'http://127.0.0.1:8000';

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file first');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setPredictions(data.predictions);
      setActiveTab('results');
      
      // Fetch additional data
      fetchMetrics();
      fetchSegmentation();
      fetchTopCustomers();
    } catch (error) {
      console.error('Error:', error);
      alert('Error uploading file. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_URL}/metrics`);
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const fetchSegmentation = async () => {
    try {
      const response = await fetch(`${API_URL}/segment`);
      const data = await response.json();
      setSegmentation(data);
    } catch (error) {
      console.error('Error fetching segmentation:', error);
    }
  };

  const fetchTopCustomers = async () => {
    try {
      const response = await fetch(`${API_URL}/topCustomers?topN=10`);
      const data = await response.json();
      setTopCustomers(data);
    } catch (error) {
      console.error('Error fetching top customers:', error);
    }
  };

  const handleExport = async () => {
    try {
      const response = await fetch(`${API_URL}/export`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'predictions.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting:', error);
      alert('Error exporting file');
    }
  };

  const handleExportHigh = async () => {
    try {
      const response = await fetch(`${API_URL}/exportHigh`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'HighValueCustomers.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting:', error);
      alert('Error exporting file');
    }
  };

  const handleExportMedium = async () => {
    try {
      const response = await fetch(`${API_URL}/exportMedium`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'MediumValueCustomers.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting:', error);
      alert('Error exporting file');
    }
  };

  const handleExportLow = async () => {
    try {
      const response = await fetch(`${API_URL}/exportLow`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'LowValueCustomers.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting:', error);
      alert('Error exporting file');
    }
  };

  const [topN, setTopN] = useState(10);

  const handleTopNChange = async () => {
    try {
      const response = await fetch(`http://localhost:8000/topCustomers?topN=${topN}`);
      const data = await response.json();
      setTopCustomers(data);
    } catch (error) {
      console.error('Error fetching top customers:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-yellow-400 text-red-600 shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Global Logistics Customer Insights</h1>
              <p className="text-xl text-red-600 mt-1">GRU RNN-Powered Customers' Future CLV Prediction</p>
            </div>
            <div className="text-red-600 px-4 py-3 font-bold text-4xl ">
              Global Logistics
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'upload'
                  ? 'border-red-600 text-red-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Upload className="inline-block w-5 h-5 mr-2" />
              Upload Data
            </button>
            <button
              onClick={() => setActiveTab('results')}
              disabled={!predictions}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'results'
                  ? 'border-red-600 text-red-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } ${!predictions ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <BarChart3 className="inline-block w-5 h-5 mr-2" />
              Results
            </button>
            <button
              onClick={() => setActiveTab('segmentation')}
              disabled={!segmentation}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'segmentation'
                  ? 'border-red-600 text-red-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } ${!segmentation ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <PieChart className="inline-block w-5 h-5 mr-2" />
              Segmentation
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-lg shadow-md p-8">
              <div className="text-center mb-8">
                <Upload className="w-16 h-16 text-red-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Customer Data</h2>
                <p className="text-gray-600">Upload a CSV file to generate predictions</p>
              </div>

              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-red-400 transition-colors">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer"
                >
                  <div className="text-gray-600">
                    <p className="text-lg mb-2">
                      {file ? file.name : 'Click to select a CSV file'}
                    </p>
                    <p className="text-sm text-gray-500">or drag and drop</p>
                  </div>
                </label>
              </div>

              <button
                onClick={handleUpload}
                disabled={!file || loading}
                className="w-full mt-6 bg-red-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors shadow-md"
              >
                {loading ? 'Processing...' : 'Generate Predictions'}
              </button>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && predictions && (
          <div className="space-y-6">
            {/* Metrics Cards */}
            {metrics && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-500 text-sm font-medium">Total Customers</p>
                      <p className="text-3xl font-bold text-gray-800 mt-1">{metrics.count}</p>
                    </div>
                    <Users className="w-12 h-12 text-red-600 opacity-80" />
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-500 text-sm font-medium">Average Value</p>
                      <p className="text-3xl font-bold text-gray-800 mt-1">{metrics.mean.toFixed(2)}</p>
                    </div>
                    <ChevronsDownUp className="w-12 h-12 text-yellow-500 opacity-80" />
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-500 text-sm font-medium">90th Percentile</p>
                      <p className="text-3xl font-bold text-gray-800 mt-1">{metrics.p90.toFixed(2)}</p>
                    </div>
                    <BarChart3 className="w-12 h-12 text-red-600 opacity-80" />
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-500 text-sm font-medium">Max Value</p>
                      <p className="text-3xl font-bold text-gray-800 mt-1">{metrics.max.toFixed(2)}</p>
                    </div>
                    <TrendingUp className="w-12 h-12 text-yellow-500 opacity-80" />
                  </div>
                </div>
              </div>
            )}

            {/* Top Customers */}
            {topCustomers && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-4">
                      <h3 className="text-xl font-bold text-gray-800">Top Customers</h3>
                      <input
                        type="number"
                        min="1"
                        max="100"
                        defaultValue="10"
                        value={topN}
                        onChange={(e) => setTopN(e.target.value)}
                        className="w-20 px-2 py-1 border border-gray-300 rounded"
                      />
                      <button
                        onClick={handleTopNChange}
                        className="bg-yellow-400 text-red-700 px-3 py-1 rounded hover:bg-yellow-500 transition-colors text-sm font-semibold"
                      >
                        Update
                      </button>
                    </div>
                    <button
                      onClick={handleExport}
                      className="bg-yellow-400 text-red-700 py-2 px-4 rounded-lg font-semibold hover:bg-yellow-500 transition-colors flex items-center shadow-md"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Export CSV
                    </button>
                  </div>

                  <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
                    <table className="w-full">
                      <thead className="sticky top-0 bg-white">
                        <tr className="border-b-2 border-gray-200">
                          <th className="text-left py-3 px-4 text-gray-600 font-semibold">Rank</th>
                          <th className="text-left py-3 px-4 text-gray-600 font-semibold">Customer ID</th>
                          <th className="text-left py-3 px-4 text-gray-600 font-semibold">Predicted Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {topCustomers.map((customer, index) => (
                          <tr key={customer.CustomerID} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="py-3 px-4">
                              <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-red-100 text-red-700 font-bold">
                                {index + 1}
                              </span>
                            </td>
                            <td className="py-3 px-4 font-medium text-gray-800">{customer.CustomerID}</td>
                            <td className="py-3 px-4">
                              <span className="inline-block bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full font-semibold">
                                {customer.Prediction.toFixed(1) + ' £'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
          </div>
        )}

        {/* Segmentation Tab */}
        {activeTab === 'segmentation' && segmentation && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-6">Customer Segmentation</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* High Value */}
                <div className="border-2 border-red-200 rounded-lg p-6 bg-red-50">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-red-800">High Value</h4>
                    <div className="w-12 h-12 rounded-full bg-red-600 flex items-center justify-center">
                      <Users className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <p className="text-4xl font-bold text-red-700 mb-2">
                    {segmentation.segmentation.high_value.count}
                  </p>
                  <p className="text-sm text-red-600">
                    {segmentation.segmentation.high_value.threshold}
                  </p>
                  <p className="text-xs text-gray-600 mt-2">
                    {((segmentation.segmentation.high_value.count / metrics.count) * 100).toFixed(1)}% of total
                  </p>
                  <button
                    onClick={handleExportHigh}
                    className="mt-4 ml-auto bg-yellow-400 text-red-700 py-2 px-4 rounded-lg font-semibold hover:bg-yellow-500 transition-colors flex items-center shadow-md"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Export CSV
                  </button>
                </div>

                {/* Medium Value */}
                <div className="border-2 border-yellow-200 rounded-lg p-6 bg-yellow-50">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-yellow-800">Medium Value</h4>
                    <div className="w-12 h-12 rounded-full bg-yellow-500 flex items-center justify-center">
                      <Users className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <p className="text-4xl font-bold text-yellow-700 mb-2">
                    {segmentation.segmentation.medium_value.count}
                  </p>
                  <p className="text-sm text-yellow-600">
                    {segmentation.segmentation.medium_value.threshold}
                  </p>
                  <p className="text-xs text-gray-600 mt-2">
                    {((segmentation.segmentation.medium_value.count / metrics.count) * 100).toFixed(1)}% of total
                  </p>
                  <button
                    onClick={handleExportMedium}
                    className="mt-4 ml-auto bg-yellow-400 text-red-700 py-2 px-4 rounded-lg font-semibold hover:bg-yellow-500 transition-colors flex items-center shadow-md"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Export CSV
                  </button>
                </div>

                {/* Low Value */}
                <div className="border-2 border-gray-200 rounded-lg p-6 bg-gray-50">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-gray-800">Low Value</h4>
                    <div className="w-12 h-12 rounded-full bg-gray-500 flex items-center justify-center">
                      <Users className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <p className="text-4xl font-bold text-gray-700 mb-2">
                    {segmentation.segmentation.low_value.count}
                  </p>
                  <p className="text-sm text-gray-600">
                    {segmentation.segmentation.low_value.threshold}
                  </p>
                  <p className="text-xs text-gray-600 mt-2">
                    {((segmentation.segmentation.low_value.count / metrics.count) * 100).toFixed(1)}% of total
                  </p>
                  <button
                    onClick={handleExportLow}
                    className="mt-4 ml-auto bg-yellow-400 text-red-700 py-2 px-4 rounded-lg font-semibold hover:bg-yellow-500 transition-colors flex items-center shadow-md"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Export CSV
                  </button>
                </div>
              </div>

              {/* Thresholds Info */}
              <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h5 className="font-semibold text-blue-900 mb-2">Segmentation Thresholds</h5>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-blue-700 font-medium">75th Percentile: </span>
                    <span className="text-blue-900">{segmentation.thresholds.p75.toFixed(1) + ' £'}</span>
                  </div>
                  <div>
                    <span className="text-blue-700 font-medium">90th Percentile: </span>
                    <span className="text-blue-900">{segmentation.thresholds.p90.toFixed(1) + ' £'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-gray-800 text-white py-7">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-gray-400">© 2024 Global Logistics Customer Insights Platform</p>
        </div>
      </footer>
    </div>
  );
};

export default App;