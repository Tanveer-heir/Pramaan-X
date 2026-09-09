import React from 'react';
import { useCaseStore } from '../../store/caseStore';
import { InvestigationEmptyState } from '../InvestigationEmptyState';
import { InvestigationOverview } from '../results/InvestigationOverview';

export const Tab2Provenance: React.FC = () => {
  const { activeDossier, dossierSource } = useCaseStore();

  if (!activeDossier) {
    return <InvestigationEmptyState title="No investigation results loaded" />;
  }

  return <InvestigationOverview investigation={activeDossier} dossierSource={dossierSource} />;
};
