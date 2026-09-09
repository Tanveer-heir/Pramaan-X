import React from 'react';
import { FileSearch } from 'lucide-react';

export const InvestigationEmptyState: React.FC<{ title?: string }> = ({
  title = 'No investigation loaded',
}) => (
  <div className="max-w-6xl mx-auto px-4 sm:px-6 py-16">
    <div className="bg-white border border-gray-200 rounded-xl p-10 text-center space-y-3">
      <div className="w-12 h-12 rounded-full bg-blue-50 text-blue-700 flex items-center justify-center mx-auto">
        <FileSearch className="w-6 h-6" />
      </div>
      <h1 className="text-lg font-semibold text-gray-950">{title}</h1>
      <p className="text-sm text-gray-600 max-w-lg mx-auto">
        Upload evidence from the Evidence Ingestion tab to populate this view with the latest gateway response. Sample exhibits are available only when selected explicitly in the header.
      </p>
    </div>
  </div>
);
